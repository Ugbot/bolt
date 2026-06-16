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

#include <cassert>
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
#  define BOLT_BAD_SOCK INVALID_SOCKET
#  define BOLT_CLOSESOCK(s) closesocket(s)
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
   using SockFd = int;
#  define BOLT_BAD_SOCK (-1)
#  define BOLT_CLOSESOCK(s) ::close(s)
#endif

namespace bolt {
namespace net {

namespace {

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
    static bool inited = false;
    if (inited) return true;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    inited = true;
    return true;
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

SockFd tcp_connect(uint32_t ipv4_host, uint16_t port) noexcept {
    SockFd fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == BOLT_BAD_SOCK) return BOLT_BAD_SOCK;
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(ipv4_host);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        BOLT_CLOSESOCK(fd);
        return BOLT_BAD_SOCK;
    }
    return fd;
}

struct HttpConn {
    bool      https;
    TlsSocket tls;
    SockFd    fd;
};

int64_t conn_send_all(HttpConn* c, const void* data, uint32_t len) noexcept {
    assert(c != nullptr);
    assert(data != nullptr);
    if (c->https) return tls_send_all(&c->tls, data, len);
    const char* p = static_cast<const char*>(data);
    uint32_t off = 0;
    while (off < len) {
        int n = static_cast<int>(send(c->fd, p + off,
                                      static_cast<int>(len - off), 0));
        if (n <= 0) return -1;
        off += static_cast<uint32_t>(n);
    }
    return static_cast<int64_t>(len);
}

int32_t conn_recv_some(HttpConn* c, void* buf, uint32_t cap) noexcept {
    assert(c != nullptr);
    assert(buf != nullptr);
    if (c->https) return tls_recv_some(&c->tls, buf, cap);
    int n = static_cast<int>(recv(c->fd, static_cast<char*>(buf),
                                  static_cast<int>(cap), 0));
    if (n < 0) return -1;
    return n;
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
    // Status line: "HTTP/1.x <code> <reason>\r\n"
    uint32_t i = 0;
    while (i < len && buf[i] != ' ') ++i;
    if (i >= len) return -1;
    ++i;
    uint32_t code_start = i;
    while (i < len && buf[i] != ' ' && buf[i] != '\r') ++i;
    if (i >= len) return -1;
    char codebuf[8];
    uint32_t clen = i - code_start;
    if (clen >= sizeof(codebuf)) return -1;
    std::memcpy(codebuf, buf + code_start, clen);
    codebuf[clen] = '\0';
    out->status = std::atoi(codebuf);
    while (i < len && buf[i] != '\n') ++i;
    if (i >= len) return -1;
    ++i;
    // Headers.
    out->header_count = 0;
    while (i < len) {
        if (i + 1 < len && buf[i] == '\r' && buf[i + 1] == '\n') {
            return static_cast<int>(i + 2);
        }
        uint32_t name_start = i;
        while (i < len && buf[i] != ':' && buf[i] != '\r') ++i;
        if (i >= len || buf[i] != ':') return -1;
        uint32_t name_end = i;
        ++i;
        while (i < len && (buf[i] == ' ' || buf[i] == '\t')) ++i;
        uint32_t val_start = i;
        while (i < len && buf[i] != '\r') ++i;
        if (i >= len) return -1;
        uint32_t val_end = i;
        if (i + 1 >= len || buf[i + 1] != '\n') return -1;
        i += 2;
        if (out->header_count < kHttpMaxHeaders) {
            HttpHeader& h = out->headers[out->header_count++];
            uint32_t nlen = name_end - name_start;
            if (nlen >= sizeof(h.name)) nlen = sizeof(h.name) - 1u;
            std::memcpy(h.name, buf + name_start, nlen);
            h.name[nlen] = '\0';
            uint32_t vlen = val_end - val_start;
            if (vlen >= sizeof(h.value)) vlen = sizeof(h.value) - 1u;
            std::memcpy(h.value, buf + val_start, vlen);
            h.value[vlen] = '\0';
        }
    }
    return -1;
}

// Recv until at least `need` bytes total in dyn; growable via realloc.
// Returns total bytes (-1 on error). buf points to malloc()'d storage caller frees.
int64_t recv_into(HttpConn* c, char** buf, uint32_t* cap, uint32_t* len,
                  uint32_t want_at_least) noexcept {
    assert(buf != nullptr);
    while (*len < want_at_least) {
        if (*len + 4096u > *cap) {
            uint32_t nc = *cap ? (*cap * 2u) : 8192u;
            while (nc < *len + 4096u) nc *= 2u;
            if (nc > kHttpMaxBody + 65536u) return -1;
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
    return static_cast<int64_t>(*len);
}

}  // namespace

bool http_url_parse(const char* url, bool* out_https, char* out_host,
                    uint32_t host_cap, uint16_t* out_port, char* out_path,
                    uint32_t path_cap) noexcept {
    assert(url != nullptr);
    assert(out_https != nullptr);
    assert(out_host != nullptr && out_port != nullptr && out_path != nullptr);
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
    // host[:port]
    const char* host_start = p;
    while (*p && *p != ':' && *p != '/') ++p;
    uint32_t hlen = static_cast<uint32_t>(p - host_start);
    if (hlen == 0u || hlen + 1u > host_cap) return false;
    std::memcpy(out_host, host_start, hlen);
    out_host[hlen] = '\0';
    if (*p == ':') {
        ++p;
        char pbuf[8];
        uint32_t pi = 0;
        while (*p && *p != '/' && pi + 1u < sizeof(pbuf)) {
            pbuf[pi++] = *p++;
        }
        pbuf[pi] = '\0';
        int port = std::atoi(pbuf);
        if (port <= 0 || port > 65535) return false;
        *out_port = static_cast<uint16_t>(port);
    }
    // path
    if (*p == '\0') {
        if (path_cap < 2u) return false;
        out_path[0] = '/';
        out_path[1] = '\0';
    } else {
        uint32_t plen = static_cast<uint32_t>(std::strlen(p));
        if (plen + 1u > path_cap) return false;
        std::memcpy(out_path, p, plen + 1u);
    }
    return true;
}

const char* http_header_get(const HttpHeader* hdrs, uint32_t n,
                            const char* name) noexcept {
    assert(hdrs != nullptr);
    assert(name != nullptr);
    return find_header(hdrs, n, name);
}

bool http_request_add_header(HttpRequest* req, const char* name,
                             const char* value) noexcept {
    assert(req != nullptr);
    assert(name != nullptr && value != nullptr);
    if (req->header_count >= kHttpMaxHeaders) return false;
    HttpHeader& h = req->headers[req->header_count];
    uint32_t nl = static_cast<uint32_t>(std::strlen(name));
    uint32_t vl = static_cast<uint32_t>(std::strlen(value));
    if (nl >= sizeof(h.name) || vl >= sizeof(h.value)) return false;
    std::memcpy(h.name, name, nl + 1u);
    std::memcpy(h.value, value, vl + 1u);
    ++req->header_count;
    return true;
}

int http_send(bolt::Arena* arena, const HttpRequest* req,
              HttpResponse* out) noexcept {
    assert(arena != nullptr);
    assert(req != nullptr && out != nullptr);
    out->status = 0;
    out->header_count = 0;
    out->body = nullptr;
    out->body_len = 0;
    if (!wsa_init_once()) return -1;

    bool https;
    char host[256];
    uint16_t port;
    char path[kHttpMaxUrl];
    if (!http_url_parse(req->url, &https, host, sizeof(host), &port, path,
                        sizeof(path))) {
        return -2;
    }

    uint32_t ipv4 = 0;
    if (!resolve_host(host, &ipv4)) return -3;

    HttpConn conn;
    conn.https = https;
    conn.fd = BOLT_BAD_SOCK;
    std::memset(&conn.tls, 0, sizeof(conn.tls));
    if (https) {
        TlsConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        // SNI = host.
        uint32_t hl = static_cast<uint32_t>(std::strlen(host));
        if (hl >= sizeof(cfg.server_name)) return -4;
        std::memcpy(cfg.server_name, host, hl + 1u);
        cfg.verify_peer = true;
        cfg.handshake_timeout_ms = req->timeout_ms ? req->timeout_ms : 10000;
        cfg.recv_timeout_ms = req->timeout_ms ? req->timeout_ms : 10000;
        if (!tls_client_connect(&conn.tls, &cfg, ipv4, port)) return -5;
    } else {
        conn.fd = tcp_connect(ipv4, port);
        if (conn.fd == BOLT_BAD_SOCK) return -6;
    }

    // Build request head: "<METHOD> <path> HTTP/1.1\r\nHost: <host>\r\n" + headers + body.
    char head[16384];
    int hlen = std::snprintf(head, sizeof(head),
                             "%s %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n",
                             req->method[0] ? req->method : "GET", path, host);
    if (hlen <= 0 || static_cast<size_t>(hlen) >= sizeof(head)) {
        conn_close(&conn);
        return -7;
    }
    for (uint32_t i = 0; i < req->header_count; ++i) {
        int add = std::snprintf(head + hlen, sizeof(head) - hlen,
                                "%s: %s\r\n", req->headers[i].name,
                                req->headers[i].value);
        if (add <= 0 || static_cast<size_t>(hlen + add) >= sizeof(head)) {
            conn_close(&conn);
            return -8;
        }
        hlen += add;
    }
    if (req->body_len > 0) {
        int add = std::snprintf(head + hlen, sizeof(head) - hlen,
                                "Content-Length: %u\r\n", req->body_len);
        if (add <= 0 || static_cast<size_t>(hlen + add) >= sizeof(head)) {
            conn_close(&conn);
            return -9;
        }
        hlen += add;
    }
    if (static_cast<size_t>(hlen + 2) >= sizeof(head)) {
        conn_close(&conn);
        return -10;
    }
    head[hlen++] = '\r';
    head[hlen++] = '\n';
    if (conn_send_all(&conn, head, static_cast<uint32_t>(hlen)) < 0) {
        conn_close(&conn);
        return -11;
    }
    if (req->body_len > 0 && req->body != nullptr) {
        if (conn_send_all(&conn, req->body, req->body_len) < 0) {
            conn_close(&conn);
            return -12;
        }
    }

    // Read response.
    char* rbuf = nullptr;
    uint32_t rcap = 0;
    uint32_t rlen = 0;
    int head_end = -1;
    while (true) {
        int64_t got = recv_into(&conn, &rbuf, &rcap, &rlen, rlen + 1u);
        if (got < 0) { std::free(rbuf); conn_close(&conn); return -13; }
        if (static_cast<uint32_t>(got) == rlen && rlen > 0) {
            // No new bytes; check head.
        }
        // Scan for \r\n\r\n.
        for (uint32_t i = (rlen > 3 ? rlen - static_cast<uint32_t>(got) : 0u);
             i + 3u < rlen; ++i) {
            if (rbuf[i] == '\r' && rbuf[i + 1] == '\n' &&
                rbuf[i + 2] == '\r' && rbuf[i + 3] == '\n') {
                head_end = static_cast<int>(i + 4u);
                break;
            }
        }
        if (head_end >= 0) break;
        if (got == 0) { std::free(rbuf); conn_close(&conn); return -14; }
        if (rlen > 65536u) { std::free(rbuf); conn_close(&conn); return -15; }
    }
    int hp = parse_head(rbuf, static_cast<uint32_t>(head_end), out);
    if (hp < 0) { std::free(rbuf); conn_close(&conn); return -16; }

    // Body: Content-Length or chunked.
    const char* cl = find_header(out->headers, out->header_count,
                                 "Content-Length");
    const char* te = find_header(out->headers, out->header_count,
                                 "Transfer-Encoding");
    uint8_t* body = nullptr;
    uint32_t body_len = 0;

    if (cl != nullptr) {
        uint32_t want = static_cast<uint32_t>(std::atoll(cl));
        if (want > kHttpMaxBody) {
            std::free(rbuf); conn_close(&conn); return -17;
        }
        if (recv_into(&conn, &rbuf, &rcap, &rlen,
                      static_cast<uint32_t>(head_end) + want) < 0) {
            std::free(rbuf); conn_close(&conn); return -18;
        }
        body = static_cast<uint8_t*>(arena->allocate(want ? want : 1u, 1));
        if (body == nullptr && want > 0) {
            std::free(rbuf); conn_close(&conn); return -19;
        }
        if (want > 0) std::memcpy(body, rbuf + head_end, want);
        body_len = want;
    } else if (te != nullptr && ieq(te, "chunked")) {
        // Buffer the whole chunked stream first.
        // Find chunk-size lines.
        // Total raw body bytes from head_end onward; keep reading until 0-chunk.
        uint32_t cursor = static_cast<uint32_t>(head_end);
        // Pre-allocate from arena progressively via temp realloc buffer.
        uint8_t* tmp = nullptr;
        uint32_t tmp_cap = 0;
        uint32_t tmp_len = 0;
        while (true) {
            // Need a line: chunk-size hex \r\n.
            int line_end = -1;
            while (true) {
                for (uint32_t i = cursor; i + 1u < rlen; ++i) {
                    if (rbuf[i] == '\r' && rbuf[i + 1] == '\n') {
                        line_end = static_cast<int>(i);
                        break;
                    }
                }
                if (line_end >= 0) break;
                int64_t got = recv_into(&conn, &rbuf, &rcap, &rlen, rlen + 1u);
                if (got < 0 || static_cast<uint32_t>(got) == rlen) {
                    std::free(tmp); std::free(rbuf); conn_close(&conn); return -20;
                }
            }
            uint32_t size = 0;
            for (uint32_t i = cursor; static_cast<int>(i) < line_end; ++i) {
                char ch = rbuf[i];
                if (ch == ';') break;
                uint32_t v;
                if (ch >= '0' && ch <= '9') v = static_cast<uint32_t>(ch - '0');
                else if (ch >= 'a' && ch <= 'f') v = static_cast<uint32_t>(ch - 'a' + 10);
                else if (ch >= 'A' && ch <= 'F') v = static_cast<uint32_t>(ch - 'A' + 10);
                else break;
                size = (size << 4) | v;
            }
            cursor = static_cast<uint32_t>(line_end + 2);
            if (size == 0) break;
            if (tmp_len + size > kHttpMaxBody) {
                std::free(tmp); std::free(rbuf); conn_close(&conn); return -21;
            }
            // Ensure rlen has size+2 more bytes from cursor.
            while (rlen < cursor + size + 2u) {
                int64_t got = recv_into(&conn, &rbuf, &rcap, &rlen, rlen + 1u);
                if (got < 0 || static_cast<uint32_t>(got) == rlen) {
                    std::free(tmp); std::free(rbuf); conn_close(&conn); return -22;
                }
            }
            if (tmp_len + size > tmp_cap) {
                uint32_t nc = tmp_cap ? tmp_cap * 2u : 8192u;
                while (nc < tmp_len + size) nc *= 2u;
                uint8_t* nb = static_cast<uint8_t*>(std::realloc(tmp, nc));
                if (nb == nullptr) {
                    std::free(tmp); std::free(rbuf); conn_close(&conn); return -23;
                }
                tmp = nb;
                tmp_cap = nc;
            }
            std::memcpy(tmp + tmp_len, rbuf + cursor, size);
            tmp_len += size;
            cursor += size + 2u;   // skip CRLF
        }
        body = static_cast<uint8_t*>(arena->allocate(tmp_len ? tmp_len : 1u, 1));
        if (body == nullptr && tmp_len > 0) {
            std::free(tmp); std::free(rbuf); conn_close(&conn); return -24;
        }
        if (tmp_len > 0) std::memcpy(body, tmp, tmp_len);
        body_len = tmp_len;
        std::free(tmp);
    } else {
        // Read until EOF.
        while (true) {
            int64_t got = recv_into(&conn, &rbuf, &rcap, &rlen, rlen + 1u);
            if (got < 0) { std::free(rbuf); conn_close(&conn); return -25; }
            if (static_cast<uint32_t>(got) == rlen) break;   // EOF
            if (rlen > kHttpMaxBody + 65536u) {
                std::free(rbuf); conn_close(&conn); return -26;
            }
        }
        uint32_t want = rlen - static_cast<uint32_t>(head_end);
        body = static_cast<uint8_t*>(arena->allocate(want ? want : 1u, 1));
        if (body == nullptr && want > 0) {
            std::free(rbuf); conn_close(&conn); return -27;
        }
        if (want > 0) std::memcpy(body, rbuf + head_end, want);
        body_len = want;
    }

    out->body     = body;
    out->body_len = body_len;
    std::free(rbuf);
    conn_close(&conn);
    return 0;
}

}  // namespace net
}  // namespace bolt
