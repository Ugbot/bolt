// bolt/src/net/bolt_http_client.cpp — bounded HTTP/1.1 client.
//
// Plain TCP via Winsock2/BSD sockets; HTTPS via bolt::net::TlsSocket. Parses
// Content-Length and chunked transfer-encoding. Caps body at kHttpMaxBody.
// Tiger Style: no exceptions, no smart pointers, fixed buffers, ≥2 asserts/fn.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/net/bolt_http_client.h"

#include "bolt/net/bolt_tls.h"
#include "bolt/net/bolt_http_response_head.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using SockFd = SOCKET;
   using SockIoSize = int;
#  define BOLT_BAD_SOCK INVALID_SOCKET
#  define BOLT_CLOSESOCK(s) closesocket(s)
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <fcntl.h>
#  include <cerrno>
#  include <unistd.h>
   using SockFd = int;
   using SockIoSize = std::size_t;
#  define BOLT_BAD_SOCK (-1)
#  define BOLT_CLOSESOCK(s) ::close(s)
#endif

namespace bolt {
namespace net {

namespace {

#include "bolt_http_client_limits.h"

constexpr uint32_t kDefaultTimeoutMs = 10000;

int64_t now_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t remaining_ms(int64_t deadline) noexcept {
    assert(deadline > 0);
    const int64_t left = deadline - now_ms();
    assert(kMaxTimeoutMs <= UINT16_MAX);
    return left <= 0 ? 0u : static_cast<uint32_t>(std::min<int64_t>(left, kMaxTimeoutMs));
}

uint32_t bounded_length(const char* value, uint32_t cap) noexcept {
    assert(value != nullptr);
    assert(cap > 0);
    for (uint32_t i = 0; i < cap; ++i) if (value[i] == '\0') return i;
    return cap;
}

bool socket_pending() noexcept {
#ifdef _WIN32
    const int error = WSAGetLastError();
    return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS || error == WSAEINTR;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS || errno == EINTR;
#endif
}

bool socket_wait(SockFd fd, bool read, int64_t deadline) noexcept {
    assert(fd != BOLT_BAD_SOCK);
    assert(deadline > 0);
#ifndef _WIN32
    if (fd >= FD_SETSIZE) return false;
#endif
    const uint32_t left = remaining_ms(deadline);
    if (left == 0) return false;
    fd_set ready;
    FD_ZERO(&ready);
    FD_SET(fd, &ready);
    timeval timeout{};
    timeout.tv_sec = static_cast<decltype(timeout.tv_sec)>(left / 1000);
    timeout.tv_usec = static_cast<decltype(timeout.tv_usec)>((left % 1000) * 1000);
#ifdef _WIN32
    const int count = 0; // ignored by Winsock; SOCKET is not an int index
#else
    const int count = fd + 1;
#endif
    return ::select(count, read ? &ready : nullptr,
                    read ? nullptr : &ready, nullptr, &timeout) > 0;
}

bool socket_nonblocking(SockFd fd) noexcept {
    assert(fd != BOLT_BAD_SOCK);
#ifdef _WIN32
    u_long mode = 1;
    return ioctlsocket(fd, FIONBIO, &mode) == 0;
#else
    const int flags = fcntl(fd, F_GETFL, 0);
    assert(O_NONBLOCK != 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

bool ieq(const char* a, const char* b) noexcept {
    assert(a != nullptr);
    assert(b != nullptr);
    while (*a && *b) {
        char ca = *a;
        char cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca + 32);
        if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb + 32);
        if (ca != cb) return false;
        ++a; ++b;
    }
    return *a == *b;
}

bool wsa_init_once() noexcept {
#ifdef _WIN32
    static const bool ready = []() noexcept {
        WSADATA wsa;
        return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
    }();
    return ready;
#else
    return true;
#endif
}

// Resolve hostname to a host-order IPv4 address.
bool resolve_host(const char* host, uint32_t* out_ipv4) noexcept {
    assert(host != nullptr);
    assert(out_ipv4 != nullptr);
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &res) != 0 || res == nullptr) {
        return false;
    }
    const auto* sa = reinterpret_cast<const sockaddr_in*>(res->ai_addr);
    uint32_t n = sa->sin_addr.s_addr;   // network byte order
    *out_ipv4 = ntohl(n);
    freeaddrinfo(res);
    return true;
}

SockFd tcp_connect(uint32_t ipv4_host, uint16_t port, int64_t deadline) noexcept {
    assert(port != 0);
    assert(deadline > 0);
    SockFd fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == BOLT_BAD_SOCK) return BOLT_BAD_SOCK;
#ifdef SO_NOSIGPIPE
    const int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) {
        BOLT_CLOSESOCK(fd);
        return BOLT_BAD_SOCK;
    }
#endif
    if (!socket_nonblocking(fd)) { BOLT_CLOSESOCK(fd); return BOLT_BAD_SOCK; }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(ipv4_host);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
    if (!socket_pending() || !socket_wait(fd, false, deadline)) {
        BOLT_CLOSESOCK(fd);
        return BOLT_BAD_SOCK;
    }
    int error = 0;
#ifdef _WIN32
    int size = sizeof(error);
#else
    socklen_t size = sizeof(error);
#endif
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &size) != 0 || error != 0) {
        BOLT_CLOSESOCK(fd);
        return BOLT_BAD_SOCK;
    }
    return fd;
}

struct HttpConn {
    bool      https;
    TlsSocket tls;
    SockFd    fd;
    int64_t   deadline;
};

int64_t conn_send_all(HttpConn* c, const void* data, uint32_t len) noexcept {
    assert(c != nullptr);
    assert(data != nullptr);
    const uint32_t left = remaining_ms(c->deadline);
    if (left == 0) return -1;
    if (c->https) return tls_send_all_timeout(&c->tls, data, len, left);
    const char* p = static_cast<const char*>(data);
    uint32_t off = 0;
    while (off < len) {
        if (!socket_wait(c->fd, false, c->deadline)) return -1;
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        int n = static_cast<int>(send(c->fd, p + off,
                                      static_cast<SockIoSize>(len - off), flags));
        if (n <= 0) return -1;
        off += static_cast<uint32_t>(n);
    }
    return static_cast<int64_t>(len);
}

int32_t conn_recv_some(HttpConn* c, void* buf, uint32_t cap) noexcept {
    assert(c != nullptr);
    assert(buf != nullptr);
    assert(cap > 0 && cap <= kHttpMaxBody + kMaxWireOverhead);
    const uint32_t left = remaining_ms(c->deadline);
    if (left == 0) return -1;
    if (c->https) return tls_recv_some_timeout(&c->tls, buf, cap, left);
    for (uint32_t attempt = 0; attempt < kMaxSocketRetries; ++attempt) {
        if (!socket_wait(c->fd, true, c->deadline)) return -1;
        const int n = static_cast<int>(recv(c->fd, static_cast<char*>(buf), static_cast<SockIoSize>(cap), 0));
        if (n >= 0) return n;
        if (!socket_pending()) return -1;
    }
    return -1;
}

void conn_close(HttpConn* c) noexcept {
    assert(c != nullptr);
    if (c->https) tls_close(&c->tls);
    if (c->fd != BOLT_BAD_SOCK) BOLT_CLOSESOCK(c->fd);
    c->fd = BOLT_BAD_SOCK;
}

// Find a header value (case-insensitive). Returns nullptr if missing.
const char* find_header(const HttpHeader* h, uint32_t n, const char* name) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        if (ieq(h[i].name, name)) return h[i].value;
    }
    return nullptr;
}

// Parse status line + headers; return bytes consumed or -1.
int parse_head(const char* buf, uint32_t len, HttpResponse* out) noexcept {
    assert(buf != nullptr);
    assert(out != nullptr);
    HttpResponseHead head{};
    if (http_response_head_parse(buf, len, &head) != HttpHeadParse::kComplete) return -1;
    out->status = static_cast<int>(head.status);
    out->header_count = 0;
    for (uint32_t i = 0; i < head.header_count; ++i) {
        const auto& field = head.headers[i];
        HttpHeader& header = out->headers[i];
        if (field.name.size >= sizeof(header.name) || field.value.size >= sizeof(header.value)) return -1;
        std::memcpy(header.name, field.name.bytes, field.name.size);
        header.name[field.name.size] = '\0';
        std::memcpy(header.value, field.value.bytes, field.value.size);
        header.value[field.value.size] = '\0';
        ++out->header_count;
    }
    return static_cast<int>(head.bytes);
}

// Recv until at least `need` bytes total in dyn; growable via realloc.
// Returns NEW bytes (0 on EOF, -1 on error), and updates len in place. A
// positive return can still be short on EOF: exact-length callers verify len.
int64_t recv_into(HttpConn* c, char** buf, uint32_t* cap, uint32_t* len,
                  uint32_t want_at_least, uint32_t limit) noexcept {
    assert(buf != nullptr && cap != nullptr && len != nullptr);
    assert(*len <= *cap && *cap <= limit);
    if (want_at_least > limit) return -1;
    const uint32_t before = *len;
    while (*len < want_at_least) {
        if (*len == *cap) {
            const uint32_t nc = std::min(limit, *cap ? *cap * 2u : kInitialReadCapacity);
            if (nc <= *cap) return -1;
            char* nb = static_cast<char*>(std::realloc(*buf, nc));
            if (nb == nullptr) return -1;
            *buf = nb;
            *cap = nc;
        }
        int32_t n = conn_recv_some(c, *buf + *len, *cap - *len);
        if (n < 0) return -1;
        if (n == 0) break;
        *len += static_cast<uint32_t>(n);
    }
    return static_cast<int64_t>(*len - before);
}

bool recv_line(HttpConn* c, char** buf, uint32_t* cap, uint32_t* len,
               uint32_t start, uint32_t* end, uint32_t limit) noexcept {
    assert(start <= *len && end != nullptr);
    assert(limit <= kHttpMaxBody + kMaxWireOverhead);
    for (uint32_t scanned = 0; scanned < kHttpResponseHeadCap; ++scanned) {
        const uint32_t at = start + scanned;
        if (*len < at + 2 && (recv_into(c, buf, cap, len, at + 2, limit) < 0 || *len < at + 2)) return false;
        if ((*buf)[at] == '\r' && (*buf)[at + 1] == '\n') { *end = at; return true; }
        if ((*buf)[at] == '\r' || (*buf)[at] == '\n') return false;
    }
    return false;
}

bool chunk_size(const char* begin, const char* end, uint32_t* out) noexcept {
    assert(begin != nullptr && out != nullptr);
    assert(begin <= end);
    uint32_t size = 0;
    const char* p = begin;
    for (; p < end && *p != ';'; ++p) {
        const char c = *p;
        uint32_t digit;
        if (c >= '0' && c <= '9') digit = static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = static_cast<uint32_t>(c - 'A' + 10);
        else return false;
        if (size > (kHttpMaxBody - digit) / 16u) return false;
        size = size * 16u + digit;
    }
    if (p == begin) return false;
    // Extensions do not change framing; bound them and reject line controls.
    for (; p < end; ++p) {
        const auto c = static_cast<unsigned char>(*p);
        if ((c < 32 && c != 9) || c == 127) return false;
    }
    *out = size;
    return true;
}

bool recv_trailers(HttpConn* c, char** buf, uint32_t* cap, uint32_t* len,
                   uint32_t cursor, uint32_t limit) noexcept {
    assert(cursor <= *len);
    assert(limit <= kHttpMaxBody + kMaxWireOverhead);
    HttpResponseHead trailers{};
    for (uint32_t count = 0; count <= kHttpMaxHeaders; ++count) {
        uint32_t end = 0;
        if (!recv_line(c, buf, cap, len, cursor, &end, limit)) return false;
        if (end == cursor) return true;
        if (!http_head_detail::line(*buf, cursor, end, trailers) ||
            trailers.has_content_length || trailers.has_transfer_encoding) return false;
        cursor = end + 2;
    }
    return false;
}

bool request_valid(const HttpRequest* req) noexcept {
    assert(req != nullptr);
    assert(kHttpMaxHeaders == kHttpResponseHeaderCap);
    if (req->header_count > kHttpMaxHeaders || req->body_len > kHttpMaxBody ||
        (req->body_len != 0 && req->body == nullptr) || req->timeout_ms > kMaxTimeoutMs ||
        req->max_response_body > kHttpMaxBody ||
        std::memchr(req->method, '\0', sizeof(req->method)) == nullptr ||
        std::memchr(req->url, '\0', sizeof(req->url)) == nullptr) return false;
    for (uint32_t i = 0; i < sizeof(req->method) && req->method[i] != '\0'; ++i)
        if (!http_head_detail::token(req->method[i])) return false;
    for (uint32_t i = 0; i < req->header_count; ++i) {
        const HttpHeader& header = req->headers[i];
        if (header.name[0] == '\0' || std::memchr(header.name, '\0', sizeof(header.name)) == nullptr ||
            std::memchr(header.value, '\0', sizeof(header.value)) == nullptr) return false;
        for (uint32_t j = 0; header.name[j] != '\0'; ++j)
            if (!http_head_detail::token(header.name[j])) return false;
        for (uint32_t j = 0; header.value[j] != '\0'; ++j) {
            const auto c = static_cast<unsigned char>(header.value[j]);
            if ((c < 32 && c != 9) || c == 127) return false;
        }
    }
    return true;
}

bool request_framing_valid(const HttpRequest* req) noexcept {
    assert(req != nullptr);
    assert(req->header_count <= kHttpMaxHeaders);
    bool length_seen = false;
    for (uint32_t i = 0; i < req->header_count; ++i) {
        const auto& field = req->headers[i];
        if (ieq(field.name, "Host") || ieq(field.name, "Connection") ||
            ieq(field.name, "Transfer-Encoding")) return false;
        if (!ieq(field.name, "Content-Length")) continue;
        if (length_seen) return false;
        length_seen = true;
        const char* at = field.value;
        while (*at == ' ' || *at == '\t') ++at;
        const char* digits = at;
        uint32_t size = 0;
        while (*at >= '0' && *at <= '9') {
            const auto digit = static_cast<uint32_t>(*at++ - '0');
            if (size > (kHttpMaxBody - digit) / 10u) return false;
            size = size * 10u + digit;
        }
        if (at == digits) return false;
        while (*at == ' ' || *at == '\t') ++at;
        if (*at != '\0' || size != req->body_len) return false;
    }
    return true;
}

bool parse_authority(const char** cursor, char* out_host, uint32_t host_cap,
                      uint16_t* out_port) noexcept {
    assert(cursor != nullptr && *cursor != nullptr);
    assert(out_host != nullptr && out_port != nullptr);
    const char* p = *cursor;
    const bool bracketed = *p == '[';
    if (bracketed) ++p;
    const char* host_start = p;
    while (*p && *p != '/' && *p != '?' && (bracketed ? *p != ']' : *p != ':')) {
        if (*p == '@' || *p == '[' || (!bracketed && *p == ']')) return false;
        ++p;
    }
    uint32_t hlen = static_cast<uint32_t>(p - host_start);
    if (hlen == 0u || hlen + 1u > host_cap) return false;
    std::memcpy(out_host, host_start, hlen);
    out_host[hlen] = '\0';
    if (bracketed) {
        if (*p != ']') return false;
        ++p;
        if (*p != ':' && *p != '/' && *p != '?' && *p != '\0') return false;
    }
    if (*p == ':') {
        ++p;
        uint32_t port = 0;
        const char* digits = p;
        while (*p && *p != '/' && *p != '?') {
            if (*p < '0' || *p > '9') return false;
            port = port * 10u + static_cast<uint32_t>(*p++ - '0');
            if (port > UINT16_MAX) return false;
        }
        if (p == digits || port == 0) return false;
        *out_port = static_cast<uint16_t>(port);
    }
    *cursor = p;
    return true;
}

}  // namespace

bool http_url_parse(const char* url, bool* out_https, char* out_host,
                    uint32_t host_cap, uint16_t* out_port, char* out_path,
                    uint32_t path_cap) noexcept {
    assert(url != nullptr);
    assert(out_https != nullptr);
    assert(out_host != nullptr && out_port != nullptr && out_path != nullptr);
    uint32_t url_len = 0;
    for (; url_len < kHttpMaxUrl && url[url_len] != '\0'; ++url_len) {
        const auto c = static_cast<unsigned char>(url[url_len]);
        if (c <= 32 || c == 127 || c == '#') return false;
    }
    if (url_len == kHttpMaxUrl) return false;
    const char* p = url;
    if (std::strncmp(p, "https://", 8) == 0) {
        *out_https = true;
        *out_port  = 443;
        p += 8;
    } else if (std::strncmp(p, "http://", 7) == 0) {
        *out_https = false;
        *out_port  = 80;
        p += 7;
    } else {
        return false;
    }
    if (!parse_authority(&p, out_host, host_cap, out_port)) return false;
    // path
    if (*p == '\0') {
        if (path_cap < 2u) return false;
        out_path[0] = '/';
        out_path[1] = '\0';
    } else {
        const uint32_t prefix = *p == '?' ? 1u : 0u;
        uint32_t plen = static_cast<uint32_t>(std::strlen(p));
        if (prefix + plen + 1u > path_cap) return false;
        if (prefix != 0) out_path[0] = '/';
        std::memcpy(out_path + prefix, p, plen + 1u);
    }
    return true;
}

bool http_url_authority(const char* url, char* out, uint32_t cap) noexcept {
    assert(url != nullptr && out != nullptr);
    assert(cap > 0);
    bool https = false;
    char host[256], path[kHttpMaxUrl];
    uint16_t port = 0;
    if (!http_url_parse(url, &https, host, sizeof(host), &port, path, sizeof(path))) return false;
    const bool ipv6 = std::strchr(host, ':') != nullptr;
    const bool standard_port = port == (https ? 443 : 80);
    const int n = standard_port
        ? std::snprintf(out, cap, ipv6 ? "[%s]" : "%s", host)
        : std::snprintf(out, cap, ipv6 ? "[%s]:%u" : "%s:%u", host, port);
    return n > 0 && static_cast<uint32_t>(n) < cap;
}

const char* http_header_get(const HttpHeader* hdrs, uint32_t n,
                            const char* name) noexcept {
    assert(hdrs != nullptr);
    assert(name != nullptr);
    if (n > kHttpMaxHeaders) return nullptr;
    return find_header(hdrs, n, name);
}

bool http_request_add_header(HttpRequest* req, const char* name,
                             const char* value) noexcept {
    assert(req != nullptr);
    assert(name != nullptr && value != nullptr);
    if (req->header_count >= kHttpMaxHeaders) return false;
    HttpHeader& h = req->headers[req->header_count];
    uint32_t nl = bounded_length(name, sizeof(h.name));
    uint32_t vl = bounded_length(value, sizeof(h.value));
    if (nl == 0 || nl >= sizeof(h.name) || vl >= sizeof(h.value)) return false;
    for (uint32_t i = 0; i < nl; ++i) if (!http_head_detail::token(name[i])) return false;
    for (uint32_t i = 0; i < vl; ++i) {
        const auto c = static_cast<unsigned char>(value[i]);
        if ((c < 32 && c != 9) || c == 127) return false;
    }
    std::memcpy(h.name, name, nl + 1u);
    std::memcpy(h.value, value, vl + 1u);
    ++req->header_count;
    return true;
}

namespace {

bool open_connection(HttpConn* c, const HttpRequest* req, bool https,
                      const char* host, uint32_t ipv4, uint16_t port) noexcept {
    assert(c != nullptr && req != nullptr);
    assert(host != nullptr && port != 0);
    *c = {};
    c->https = https;
    c->fd = BOLT_BAD_SOCK;
    c->deadline = now_ms() + (req->timeout_ms ? req->timeout_ms : kDefaultTimeoutMs);
    if (!https) {
        c->fd = tcp_connect(ipv4, port, c->deadline);
        return c->fd != BOLT_BAD_SOCK;
    }
    TlsConfig cfg{};
    const size_t size = std::strlen(host);
    if (size >= sizeof(cfg.server_name)) return false;
    std::memcpy(cfg.server_name, host, size + 1);
    cfg.verify_peer = true;
    cfg.handshake_timeout_ms = static_cast<uint16_t>(remaining_ms(c->deadline));
    return cfg.handshake_timeout_ms != 0 && tls_client_connect(&c->tls, &cfg, ipv4, port);
}

bool send_request_head(HttpConn* c, const HttpRequest* req,
                       const char* path, const char* authority) noexcept {
    assert(c != nullptr && req != nullptr);
    assert(path != nullptr && authority != nullptr);
    constexpr uint32_t kRequestHeadCap = http_client_limits::kRequestHeadCap;
    char head[kRequestHeadCap];
    int size = std::snprintf(head, sizeof(head),
        "%s %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n",
        req->method[0] ? req->method : "GET", path, authority);
    if (size <= 0 || static_cast<size_t>(size) >= sizeof(head)) return false;
    for (uint32_t i = 0; i < req->header_count; ++i) {
        if (ieq(req->headers[i].name, "Content-Length")) continue;
        const int added = std::snprintf(head + size, sizeof(head) - static_cast<size_t>(size),
            "%s: %s\r\n", req->headers[i].name, req->headers[i].value);
        if (added <= 0 || static_cast<size_t>(size + added) >= sizeof(head)) return false;
        size += added;
    }
    if (req->body_len > 0 || find_header(req->headers, req->header_count, "Content-Length") != nullptr) {
        const int added = std::snprintf(head + size, sizeof(head) - static_cast<size_t>(size),
            "Content-Length: %u\r\n", req->body_len);
        if (added <= 0 || static_cast<size_t>(size + added) >= sizeof(head)) return false;
        size += added;
    }
    if (static_cast<size_t>(size + 2) > sizeof(head)) return false;
    head[size++] = '\r';
    head[size++] = '\n';
    if (conn_send_all(c, head, static_cast<uint32_t>(size)) < 0) return false;
    return req->body_len == 0 || conn_send_all(c, req->body, req->body_len) >= 0;
}

struct ResponseRead {
    HttpConn* conn;
    Arena* arena;
    HttpResponse* out;
    char* data{nullptr};
    uint32_t capacity{0}, size{0}, head_end{0};
    uint32_t body_limit, wire_limit;
};

bool response_head(ResponseRead* r) noexcept {
    assert(r != nullptr && r->conn != nullptr);
    assert(r->body_limit <= kHttpMaxBody);
    for (uint32_t reads = 0; reads < kHttpResponseHeadCap; ++reads) {
        const uint32_t before = r->size;
        if (recv_into(r->conn, &r->data, &r->capacity, &r->size, before + 1, r->wire_limit) <= 0) return false;
        for (uint32_t i = before > 3 ? before - 3 : 0; i + 3 < r->size; ++i) {
            if (std::memcmp(r->data + i, "\r\n\r\n", 4) != 0) continue;
            r->head_end = i + 4;
            return parse_head(r->data, r->head_end, r->out) >= 0;
        }
        if (r->size >= kHttpResponseHeadCap) return false;
    }
    return false;
}

bool publish_body(ResponseRead* r, const void* bytes, uint32_t size) noexcept {
    assert(r != nullptr && r->out->body == nullptr);
    assert(bytes != nullptr || size == 0);
    if (size > r->body_limit) return false;
    if (size == 0) return true;
    auto* body = static_cast<uint8_t*>(r->arena->allocate(size, 1));
    if (body == nullptr) return false;
    std::memcpy(body, bytes, size);
    r->out->body = body;
    r->out->body_len = size;
    return true;
}

bool response_length(ResponseRead* r, const char* length) noexcept {
    assert(r != nullptr && length != nullptr);
    assert(r->head_end <= r->size);
    // The shared head parser validated decimal syntax and uint64 overflow.
    const uint64_t declared = std::strtoull(length, nullptr, 10);
    if (declared > r->body_limit) return false;
    const uint32_t size = static_cast<uint32_t>(declared);
    const uint32_t need = r->head_end + size;
    if (recv_into(r->conn, &r->data, &r->capacity, &r->size, need, r->wire_limit) < 0 ||
        r->size < need) return false;
    return publish_body(r, r->data + r->head_end, size);
}

struct DecodedBody {
    uint8_t* data{nullptr};
    uint32_t capacity{0}, size{0};
};

bool append_chunk(ResponseRead* r, DecodedBody* body, uint32_t cursor, uint32_t size) noexcept {
    assert(r != nullptr && body != nullptr);
    assert(body->size <= r->body_limit);
    if (size > r->body_limit - body->size) return false;
    const uint32_t need = cursor + size + 2;
    if (recv_into(r->conn, &r->data, &r->capacity, &r->size, need, r->wire_limit) < 0 ||
        r->size < need || r->data[need - 2] != '\r' || r->data[need - 1] != '\n') return false;
    const uint32_t total = body->size + size;
    if (total > body->capacity) {
        uint32_t capacity = body->capacity ? body->capacity : kInitialReadCapacity;
        while (capacity < total) capacity *= 2;
        capacity = std::min(capacity, r->body_limit);
        auto* data = static_cast<uint8_t*>(std::realloc(body->data, capacity));
        if (data == nullptr) return false;
        body->data = data;
        body->capacity = capacity;
    }
    std::memcpy(body->data + body->size, r->data + cursor, size);
    body->size = total;
    return true;
}

bool response_chunked(ResponseRead* r) noexcept {
    assert(r != nullptr && r->conn != nullptr);
    assert(r->head_end <= r->size);
    DecodedBody body;
    uint32_t cursor = r->head_end;
    bool complete = false;
    // Every nonterminal chunk contributes at least one body byte.
    for (uint32_t chunks = 0; chunks <= r->body_limit; ++chunks) {
        uint32_t end = 0, size = 0;
        if (!recv_line(r->conn, &r->data, &r->capacity, &r->size, cursor, &end, r->wire_limit) ||
            !chunk_size(r->data + cursor, r->data + end, &size)) break;
        cursor = end + 2;
        if (size == 0) {
            complete = recv_trailers(r->conn, &r->data, &r->capacity, &r->size, cursor, r->wire_limit) &&
                publish_body(r, body.data, body.size);
            break;
        }
        if (!append_chunk(r, &body, cursor, size)) break;
        cursor += size + 2;
    }
    std::free(body.data);
    return complete;
}

bool response_eof(ResponseRead* r) noexcept {
    assert(r != nullptr && r->conn != nullptr);
    assert(r->head_end <= r->size);
    // Each positive read consumes wire budget; zero is clean EOF.
    for (uint32_t reads = 0; reads <= r->wire_limit; ++reads) {
        if (r->size - r->head_end > r->body_limit) return false;
        const int64_t got = recv_into(r->conn, &r->data, &r->capacity, &r->size, r->size + 1, r->wire_limit);
        if (got < 0) return false;
        if (got == 0) return publish_body(r, r->data + r->head_end, r->size - r->head_end);
    }
    return false;
}

bool receive_response(HttpConn* c, const HttpRequest* req, Arena* arena, HttpResponse* out) noexcept {
    assert(c != nullptr && req != nullptr);
    assert(arena != nullptr && out != nullptr);
    const uint32_t limit = req->max_response_body ? req->max_response_body : kHttpMaxBody;
    ResponseRead read{c, arena, out, nullptr, 0, 0, 0, limit, limit + kMaxWireOverhead};
    bool complete = response_head(&read);
    if (complete) {
        const char* cl = find_header(out->headers, out->header_count, "Content-Length");
        const char* te = find_header(out->headers, out->header_count, "Transfer-Encoding");
        const bool no_body = ieq(req->method, "HEAD") || out->status == 204 ||
            out->status == 304 || out->status / 100 == 1;
        if (!no_body) {
            if (cl != nullptr) complete = response_length(&read, cl);
            else if (te != nullptr) complete = ieq(te, "chunked") && response_chunked(&read);
            else complete = response_eof(&read);
        }
    }
    std::free(read.data);
    return complete;
}

} // namespace

int http_send(bolt::Arena* arena, const HttpRequest* req, HttpResponse* out) noexcept {
    assert(arena != nullptr);
    assert(req != nullptr && out != nullptr);
    out->status = 0;
    out->header_count = 0;
    out->body = nullptr;
    out->body_len = 0;
    if (!request_valid(req) || !request_framing_valid(req) || !wsa_init_once()) return -1;
    bool https = false;
    char host[256], path[kHttpMaxUrl], authority[sizeof(host) + 8];
    uint16_t port = 0;
    if (!http_url_parse(req->url, &https, host, sizeof(host), &port, path, sizeof(path)) ||
        !http_url_authority(req->url, authority, sizeof(authority))) return -2;
    uint32_t ipv4 = 0;
    if (!resolve_host(host, &ipv4)) return -3;
    HttpConn connection;
    if (!open_connection(&connection, req, https, host, ipv4, port)) return -5;
    const bool sent = send_request_head(&connection, req, path, authority);
    const bool complete = sent && receive_response(&connection, req, arena, out);
    conn_close(&connection);
    return complete ? 0 : -1;
}

} // namespace net
} // namespace bolt
