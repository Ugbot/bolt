// bolt/crypto/sigv4.cpp — AWS SigV4 + a self-contained SHA-256 / HMAC-SHA256.
// Zero external dependency (no OpenSSL): bolt stays a leaf library.

#include "bolt/crypto/sigv4.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace crypto {

namespace {

// ---- SHA-256 (FIPS 180-4) -------------------------------------------------

struct Sha256Ctx {
    uint32_t h[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    uint32_t buflen;
};

inline uint32_t rotr(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32u - n));
}

const uint32_t kK[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
    0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
    0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
    0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
    0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
    0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

void sha256_init(Sha256Ctx* c) noexcept {
    assert(c != nullptr);
    c->h[0]=0x6a09e667u; c->h[1]=0xbb67ae85u; c->h[2]=0x3c6ef372u;
    c->h[3]=0xa54ff53au; c->h[4]=0x510e527fu; c->h[5]=0x9b05688cu;
    c->h[6]=0x1f83d9abu; c->h[7]=0x5be0cd19u;
    c->bitlen = 0;
    c->buflen = 0;
}

void sha256_block(Sha256Ctx* c, const uint8_t* p) noexcept {
    assert(c != nullptr);
    assert(p != nullptr);
    uint32_t w[64];
    for (uint32_t i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(p[4*i]) << 24) |
               (static_cast<uint32_t>(p[4*i+1]) << 16) |
               (static_cast<uint32_t>(p[4*i+2]) << 8) |
                static_cast<uint32_t>(p[4*i+3]);
    }
    for (uint32_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        const uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=c->h[0],b=c->h[1],cc=c->h[2],d=c->h[3];
    uint32_t e=c->h[4],f=c->h[5],g=c->h[6],hh=c->h[7];
    for (uint32_t i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
        const uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        const uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        const uint32_t t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    c->h[0]+=a; c->h[1]+=b; c->h[2]+=cc; c->h[3]+=d;
    c->h[4]+=e; c->h[5]+=f; c->h[6]+=g; c->h[7]+=hh;
}

void sha256_update(Sha256Ctx* c, const uint8_t* data, uint64_t len) noexcept {
    assert(c != nullptr);
    assert(data != nullptr || len == 0);
    c->bitlen += len * 8u;
    uint64_t i = 0;
    while (i < len) {                              // bounded by len
        const uint32_t space = 64u - c->buflen;
        uint32_t take = static_cast<uint32_t>(len - i);
        if (take > space) take = space;
        std::memcpy(c->buf + c->buflen, data + i, take);
        c->buflen += take;
        i += take;
        if (c->buflen == 64u) {
            sha256_block(c, c->buf);
            c->buflen = 0;
        }
    }
}

void sha256_final(Sha256Ctx* c, uint8_t out[32]) noexcept {
    assert(c != nullptr);
    assert(out != nullptr);
    const uint64_t bits = c->bitlen;
    uint8_t pad = 0x80u;
    sha256_update(c, &pad, 1);
    pad = 0x00u;
    while (c->buflen != 56u) {                     // bounded: <= 64 iters
        sha256_update(c, &pad, 1);
    }
    uint8_t lenbuf[8];
    for (uint32_t i = 0; i < 8; ++i) {
        lenbuf[i] = static_cast<uint8_t>((bits >> (56u - 8u*i)) & 0xFFu);
    }
    sha256_update(c, lenbuf, 8);
    for (uint32_t i = 0; i < 8; ++i) {
        out[4*i]   = static_cast<uint8_t>((c->h[i] >> 24) & 0xFFu);
        out[4*i+1] = static_cast<uint8_t>((c->h[i] >> 16) & 0xFFu);
        out[4*i+2] = static_cast<uint8_t>((c->h[i] >> 8) & 0xFFu);
        out[4*i+3] = static_cast<uint8_t>(c->h[i] & 0xFFu);
    }
}

void hex_lower(const uint8_t* d, uint32_t n, char* out) noexcept {
    assert(d != nullptr);
    assert(out != nullptr);
    static const char* kHex = "0123456789abcdef";
    for (uint32_t i = 0; i < n; ++i) {
        out[2*i]   = kHex[(d[i] >> 4) & 0xFu];
        out[2*i+1] = kHex[d[i] & 0xFu];
    }
    out[2*n] = '\0';
}

// Bounded string append helper. Returns false on overflow.
bool app(char* dst, uint32_t cap, uint32_t* pos, const char* s) noexcept {
    assert(dst != nullptr);
    assert(pos != nullptr);
    if (s == nullptr) return true;
    const size_t sl = std::strlen(s);
    if (static_cast<uint64_t>(*pos) + sl + 1u > cap) return false;
    std::memcpy(dst + *pos, s, sl);
    *pos += static_cast<uint32_t>(sl);
    dst[*pos] = '\0';
    return true;
}

}  // namespace

void sha256(const uint8_t* data, uint64_t len, uint8_t out[32]) noexcept {
    assert(data != nullptr || len == 0);
    assert(out != nullptr);
    Sha256Ctx c;
    sha256_init(&c);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}

void hmac_sha256(const uint8_t* key, uint32_t key_len,
                 const uint8_t* msg, uint64_t msg_len,
                 uint8_t out[32]) noexcept {
    assert(key != nullptr || key_len == 0);
    assert(out != nullptr);
    uint8_t k[64];
    std::memset(k, 0, sizeof(k));
    if (key_len > 64u) {
        sha256(key, key_len, k);                   // long keys are hashed
    } else if (key_len > 0u) {
        std::memcpy(k, key, key_len);
    }
    uint8_t ipad[64], opad[64];
    for (uint32_t i = 0; i < 64; ++i) {
        ipad[i] = static_cast<uint8_t>(k[i] ^ 0x36u);
        opad[i] = static_cast<uint8_t>(k[i] ^ 0x5cu);
    }
    uint8_t inner[32];
    Sha256Ctx c;
    sha256_init(&c);
    sha256_update(&c, ipad, 64);
    sha256_update(&c, msg, msg_len);
    sha256_final(&c, inner);
    sha256_init(&c);
    sha256_update(&c, opad, 64);
    sha256_update(&c, inner, 32);
    sha256_final(&c, out);
}

void sha256_hex(const uint8_t* data, uint64_t len, char out_hex[65]) noexcept {
    assert(data != nullptr || len == 0);
    assert(out_hex != nullptr);
    uint8_t d[32];
    sha256(data, len, d);
    hex_lower(d, 32, out_hex);
}

void sha256_empty_hex(char out_hex[65]) noexcept {
    assert(out_hex != nullptr);
    sha256_hex(reinterpret_cast<const uint8_t*>(""), 0, out_hex);
}

namespace {

// Build the canonical request into `buf`; returns its length or 0 on overflow.
// Also fills `signed_headers` (caller buffer >= 96 bytes).
uint32_t build_canonical(const SigV4Request* r, char* buf, uint32_t cap,
                         char* signed_headers, uint32_t sh_cap) noexcept {
    assert(r != nullptr);
    assert(buf != nullptr);
    uint32_t shp = 0;
    if (!app(signed_headers, sh_cap, &shp,
             "host;x-amz-content-sha256;x-amz-date")) return 0;
    if (r->session_token != nullptr) {
        if (!app(signed_headers, sh_cap, &shp, ";x-amz-security-token"))
            return 0;
    }
    uint32_t p = 0;
    bool ok = app(buf, cap, &p, r->method) && app(buf, cap, &p, "\n") &&
              app(buf, cap, &p, r->path) && app(buf, cap, &p, "\n") &&
              app(buf, cap, &p, r->query_string) && app(buf, cap, &p, "\n") &&
              app(buf, cap, &p, "host:") && app(buf, cap, &p, r->host) &&
              app(buf, cap, &p, "\nx-amz-content-sha256:") &&
              app(buf, cap, &p, r->payload_sha256) &&
              app(buf, cap, &p, "\nx-amz-date:") &&
              app(buf, cap, &p, r->amz_date) && app(buf, cap, &p, "\n");
    if (r->session_token != nullptr) {
        ok = ok && app(buf, cap, &p, "x-amz-security-token:") &&
             app(buf, cap, &p, r->session_token) && app(buf, cap, &p, "\n");
    }
    ok = ok && app(buf, cap, &p, "\n") &&
         app(buf, cap, &p, signed_headers) && app(buf, cap, &p, "\n") &&
         app(buf, cap, &p, r->payload_sha256);
    return ok ? p : 0;
}

bool required(const SigV4Request* r) noexcept {
    return r != nullptr && r->method && r->host && r->path && r->service &&
           r->region && r->amz_date && r->payload_sha256 && r->access_key &&
           r->secret_key;
}

}  // namespace

bool sigv4_sign(const SigV4Request* req, SigV4Result* out) noexcept {
    assert(req != nullptr);
    assert(out != nullptr);
    if (!required(req) || out == nullptr) return false;
    if (out->auth_header == nullptr || out->amz_date_out == nullptr)
        return false;
    // Short date = first 8 chars of amz_date ("YYYYMMDD").
    if (std::strlen(req->amz_date) < 16u) return false;
    char short_date[9];
    std::memcpy(short_date, req->amz_date, 8);
    short_date[8] = '\0';

    // Echo the date.
    if (std::strlen(req->amz_date) + 1u > out->amz_date_cap) return false;
    std::strcpy(out->amz_date_out, req->amz_date);

    // 1) canonical request → its SHA-256 hex.
    char canon[kSigV4MaxPath + kSigV4MaxQuery + kSigV4MaxHost +
               kSigV4MaxToken + 512u];
    char signed_headers[96];
    const uint32_t clen = build_canonical(req, canon, sizeof(canon),
                                          signed_headers, sizeof(signed_headers));
    if (clen == 0) return false;
    char canon_hex[65];
    sha256_hex(reinterpret_cast<const uint8_t*>(canon), clen, canon_hex);

    // 2) string to sign.
    char scope[kSigV4MaxRegion + kSigV4MaxService + 64u];
    uint32_t sp = 0;
    if (!app(scope, sizeof(scope), &sp, short_date) ||
        !app(scope, sizeof(scope), &sp, "/") ||
        !app(scope, sizeof(scope), &sp, req->region) ||
        !app(scope, sizeof(scope), &sp, "/") ||
        !app(scope, sizeof(scope), &sp, req->service) ||
        !app(scope, sizeof(scope), &sp, "/aws4_request")) return false;

    char sts[256 + sizeof(scope)];
    uint32_t tp = 0;
    if (!app(sts, sizeof(sts), &tp, "AWS4-HMAC-SHA256\n") ||
        !app(sts, sizeof(sts), &tp, req->amz_date) ||
        !app(sts, sizeof(sts), &tp, "\n") ||
        !app(sts, sizeof(sts), &tp, scope) ||
        !app(sts, sizeof(sts), &tp, "\n") ||
        !app(sts, sizeof(sts), &tp, canon_hex)) return false;

    // 3) derive the signing key: kSecret → kDate → kRegion → kService → kSign.
    char secret0[4 + kSigV4MaxSecret];
    uint32_t kp = 0;
    if (!app(secret0, sizeof(secret0), &kp, "AWS4") ||
        !app(secret0, sizeof(secret0), &kp, req->secret_key)) return false;
    uint8_t k_date[32], k_region[32], k_service[32], k_signing[32];
    hmac_sha256(reinterpret_cast<const uint8_t*>(secret0), kp,
                reinterpret_cast<const uint8_t*>(short_date),
                std::strlen(short_date), k_date);
    hmac_sha256(k_date, 32, reinterpret_cast<const uint8_t*>(req->region),
                std::strlen(req->region), k_region);
    hmac_sha256(k_region, 32, reinterpret_cast<const uint8_t*>(req->service),
                std::strlen(req->service), k_service);
    hmac_sha256(k_service, 32,
                reinterpret_cast<const uint8_t*>("aws4_request"), 12, k_signing);

    // 4) signature = HMAC(kSigning, stringToSign).
    uint8_t sig[32];
    hmac_sha256(k_signing, 32, reinterpret_cast<const uint8_t*>(sts), tp, sig);
    char sig_hex[65];
    hex_lower(sig, 32, sig_hex);

    // 5) Authorization header value.
    uint32_t ap = 0;
    char* a = out->auth_header;
    const uint32_t acap = out->auth_cap;
    bool ok = app(a, acap, &ap, "AWS4-HMAC-SHA256 Credential=") &&
              app(a, acap, &ap, req->access_key) &&
              app(a, acap, &ap, "/") && app(a, acap, &ap, scope) &&
              app(a, acap, &ap, ", SignedHeaders=") &&
              app(a, acap, &ap, signed_headers) &&
              app(a, acap, &ap, ", Signature=") &&
              app(a, acap, &ap, sig_hex);
    return ok;
}

}  // namespace crypto
}  // namespace bolt
