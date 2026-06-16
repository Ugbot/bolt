// bolt/net/bolt_tls.cpp — OpenSSL 3.x TLS 1.3 client wrapper.
//
// All platform + OpenSSL includes live in this TU only; the header is
// portable. Windows uses Winsock; POSIX uses standard sockets.
#include "bolt/net/bolt_tls.h"

#include <cassert>
#include <cstring>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
   using sock_t = SOCKET;
   static constexpr sock_t kInvalidSock = INVALID_SOCKET;
#  define BOLT_TLS_CLOSESOCK(s) ::closesocket(s)
#else
#  include <sys/socket.h>
#  include <sys/select.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <errno.h>
   using sock_t = int;
   static constexpr sock_t kInvalidSock = -1;
#  define BOLT_TLS_CLOSESOCK(s) ::close(s)
#endif

#if defined(BOLT_WITH_TLS)
#  include <openssl/ssl.h>
#  include <openssl/err.h>
#  include <openssl/x509v3.h>
#endif

#include <chrono>

namespace bolt::net {

namespace {

// ---------------------------------------------------------------------------
// Socket helpers — bounded, portable, no allocation.
// ---------------------------------------------------------------------------

// One-shot Winsock init guard. Lazy; init-on-first-connect is fine because
// the runtime serialises through atomic_flag.
#if defined(_WIN32)
struct WinsockInit {
    WinsockInit() noexcept {
        WSADATA d;
        ::WSAStartup(MAKEWORD(2, 2), &d);
    }
};
static void ensure_winsock() noexcept {
    static WinsockInit init;
    (void)init;
}
#else
static void ensure_winsock() noexcept {}
#endif

static bool set_nonblocking(sock_t fd) noexcept {
    assert(fd != kInvalidSock && "set_nonblocking: invalid fd");
#if defined(_WIN32)
    u_long mode = 1;
    return ::ioctlsocket(fd, FIONBIO, &mode) == 0;
#else
    int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl < 0) return false;
    return ::fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
#endif
}

static int last_sock_err() noexcept {
#if defined(_WIN32)
    return ::WSAGetLastError();
#else
    return errno;
#endif
}

static bool sock_in_progress(int err) noexcept {
#if defined(_WIN32)
    return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS;
#else
    return err == EINPROGRESS || err == EAGAIN || err == EWOULDBLOCK;
#endif
}

// Wait until `fd` is readable / writable, bounded by deadline_ms (epoch
// milliseconds). Returns true if ready, false on timeout or error.
static bool sock_wait(sock_t fd, bool want_read, bool want_write,
                      std::int64_t deadline_ms) noexcept {
    assert(fd != kInvalidSock && "sock_wait: invalid fd");
    assert((want_read || want_write) && "sock_wait: must want r or w");

    using clock = std::chrono::steady_clock;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now().time_since_epoch()).count();
    std::int64_t remain = deadline_ms - now_ms;
    if (remain <= 0) return false;

    fd_set rfds, wfds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    if (want_read)  FD_SET(fd, &rfds);
    if (want_write) FD_SET(fd, &wfds);

    timeval tv;
    tv.tv_sec  = static_cast<long>(remain / 1000);
    tv.tv_usec = static_cast<long>((remain % 1000) * 1000);

    const int rc = ::select(static_cast<int>(fd) + 1,
                            want_read ? &rfds : nullptr,
                            want_write ? &wfds : nullptr,
                            nullptr, &tv);
    return rc > 0;
}

static std::int64_t deadline_from(std::uint16_t ms_in, std::uint16_t fallback) noexcept {
    const std::uint16_t ms = ms_in ? ms_in : fallback;
    using clock = std::chrono::steady_clock;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now().time_since_epoch()).count();
    return now_ms + static_cast<std::int64_t>(ms);
}

// Connect a non-blocking TCP socket to (ipv4 in host order, port).
// Returns kInvalidSock on failure; on success, the returned socket is
// non-blocking and connected.
static sock_t tcp_connect(std::uint32_t ipv4_host, std::uint16_t port,
                          std::int64_t deadline_ms) noexcept {
    ensure_winsock();
    sock_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == kInvalidSock) return kInvalidSock;
    if (!set_nonblocking(fd)) {
        BOLT_TLS_CLOSESOCK(fd);
        return kInvalidSock;
    }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = htonl(ipv4_host);

    const int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (rc == 0) return fd;  // immediate connect (rare)
    if (!sock_in_progress(last_sock_err())) {
        BOLT_TLS_CLOSESOCK(fd);
        return kInvalidSock;
    }
    if (!sock_wait(fd, /*want_read*/false, /*want_write*/true, deadline_ms)) {
        BOLT_TLS_CLOSESOCK(fd);
        return kInvalidSock;
    }
    // Check SO_ERROR — connect can succeed-then-fail (e.g. RST).
    int so_err = 0;
#if defined(_WIN32)
    int slen = sizeof(so_err);
#else
    socklen_t slen = sizeof(so_err);
#endif
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR,
                     reinterpret_cast<char*>(&so_err), &slen) < 0 || so_err != 0) {
        BOLT_TLS_CLOSESOCK(fd);
        return kInvalidSock;
    }
    return fd;
}

}  // namespace

// ===========================================================================
// BOLT_WITH_TLS=OFF — link-only stubs so consumers still compile.
// ===========================================================================
#if !defined(BOLT_WITH_TLS)

bool tls_client_connect(TlsSocket* out, const TlsConfig*, std::uint32_t,
                        std::uint16_t) noexcept {
    if (out) std::memset(out, 0, sizeof(*out));
    return false;
}
std::int64_t tls_send_all(const TlsSocket*, const void*, std::uint32_t) noexcept { return -1; }
bool         tls_recv_all(const TlsSocket*, void*, std::uint32_t) noexcept       { return false; }
std::int32_t tls_recv_some(const TlsSocket*, void*, std::uint32_t) noexcept      { return -1; }
void         tls_close(TlsSocket* s) noexcept {
    if (s) std::memset(s, 0, sizeof(*s));
}

#else  // BOLT_WITH_TLS

namespace {

// Default timeouts (ms) when caller passes 0.
static constexpr std::uint16_t kDefaultHandshakeMs = 5000;
static constexpr std::uint16_t kDefaultRecvMs      = 5000;

// Configure an SSL_CTX from the caller's TlsConfig. Returns nullptr on
// failure (e.g. cert/key load error).
static SSL_CTX* make_ctx(const TlsConfig* cfg) noexcept {
    assert(cfg != nullptr && "make_ctx: null cfg");
    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return nullptr;

    // TLS 1.3 only. Fail-fast on anything less.
    if (!SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION) ||
        !SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION)) {
        SSL_CTX_free(ctx);
        return nullptr;
    }

    if (cfg->verify_peer) {
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
    } else {
        // insecure_skip_verify is the only path to VERIFY_NONE.
        SSL_CTX_set_verify(ctx,
                           cfg->insecure_skip_verify ? SSL_VERIFY_NONE
                                                     : SSL_VERIFY_PEER,
                           nullptr);
    }

    // Trust store: explicit ca_path overrides; empty string → system store.
    if (cfg->ca_path[0] != '\0') {
        if (SSL_CTX_load_verify_locations(ctx, cfg->ca_path, nullptr) != 1) {
            SSL_CTX_free(ctx);
            return nullptr;
        }
    } else {
        // System defaults: on Windows OpenSSL 3.x picks up the user's
        // configured root store; on POSIX it walks /etc/ssl/certs and
        // friends. Both are fine for client TLS.
        if (SSL_CTX_set_default_verify_paths(ctx) != 1) {
            SSL_CTX_free(ctx);
            return nullptr;
        }
    }

    // mTLS: load client cert + key when both paths are set.
    if (cfg->client_cert_path[0] != '\0' && cfg->client_key_path[0] != '\0') {
        if (SSL_CTX_use_certificate_chain_file(ctx, cfg->client_cert_path) != 1 ||
            SSL_CTX_use_PrivateKey_file(ctx, cfg->client_key_path, SSL_FILETYPE_PEM) != 1 ||
            SSL_CTX_check_private_key(ctx) != 1) {
            SSL_CTX_free(ctx);
            return nullptr;
        }
    }
    return ctx;
}

// Drive a single SSL_connect call inside the bounded handshake loop.
// Returns 1 on success, 0 to keep looping, -1 on hard failure.
static int handshake_step(SSL* ssl, sock_t fd, std::int64_t deadline_ms) noexcept {
    assert(ssl != nullptr && "handshake_step: null ssl");
    assert(fd  != kInvalidSock && "handshake_step: invalid fd");

    const int rc = SSL_connect(ssl);
    if (rc == 1) return 1;
    const int err = SSL_get_error(ssl, rc);
    if (err == SSL_ERROR_WANT_READ) {
        return sock_wait(fd, /*r*/true, /*w*/false, deadline_ms) ? 0 : -1;
    }
    if (err == SSL_ERROR_WANT_WRITE) {
        return sock_wait(fd, /*r*/false, /*w*/true, deadline_ms) ? 0 : -1;
    }
    return -1;
}

}  // namespace

bool tls_client_connect(TlsSocket* out, const TlsConfig* cfg,
                        std::uint32_t ipv4, std::uint16_t port) noexcept {
    assert(out != nullptr && "tls_client_connect: null out");
    assert(cfg != nullptr && "tls_client_connect: null cfg");

    std::memset(out, 0, sizeof(*out));
    out->fd = static_cast<int>(kInvalidSock);

    const std::int64_t hs_deadline =
        deadline_from(cfg->handshake_timeout_ms, kDefaultHandshakeMs);

    sock_t fd = tcp_connect(ipv4, port, hs_deadline);
    if (fd == kInvalidSock) return false;

    SSL_CTX* ctx = make_ctx(cfg);
    if (!ctx) { BOLT_TLS_CLOSESOCK(fd); return false; }

    SSL* ssl = SSL_new(ctx);
    if (!ssl) { SSL_CTX_free(ctx); BOLT_TLS_CLOSESOCK(fd); return false; }

    if (SSL_set_fd(ssl, static_cast<int>(fd)) != 1) {
        SSL_free(ssl); SSL_CTX_free(ctx); BOLT_TLS_CLOSESOCK(fd); return false;
    }
    // SNI — required by all modern HTTPS endpoints.
    if (cfg->server_name[0] != '\0') {
        SSL_set_tlsext_host_name(ssl, cfg->server_name);
        // Hostname verification (RFC 6125) at the X509 layer.
        if (cfg->verify_peer) {
            X509_VERIFY_PARAM* p = SSL_get0_param(ssl);
            X509_VERIFY_PARAM_set_hostflags(p, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
            X509_VERIFY_PARAM_set1_host(p, cfg->server_name, 0);
        }
    }

    // Bounded handshake loop. Each spin either WANTs r/w (and we select)
    // or finishes / fails.
    for (int spins = 0; spins < 256; ++spins) {
        const int s = handshake_step(ssl, fd, hs_deadline);
        if (s == 1) {
            out->ssl_ctx        = ctx;
            out->ssl            = ssl;
            out->fd             = static_cast<int>(fd);
            out->handshake_done = true;
            return true;
        }
        if (s < 0) break;  // timeout / hard error
    }
    SSL_free(ssl); SSL_CTX_free(ctx); BOLT_TLS_CLOSESOCK(fd);
    return false;
}

std::int64_t tls_send_all(const TlsSocket* s, const void* data,
                          std::uint32_t len) noexcept {
    assert(s != nullptr && "tls_send_all: null s");
    assert(data != nullptr && "tls_send_all: null data");
    if (!s->handshake_done || !s->ssl) return -1;
    SSL* ssl = static_cast<SSL*>(s->ssl);
    const sock_t fd = static_cast<sock_t>(s->fd);

    // Use a fresh deadline per call (recv_timeout_ms doubles as send deadline).
    // We can't reach cfg here — pick the library default.
    const std::int64_t deadline = deadline_from(0, kDefaultRecvMs);

    std::uint32_t off = 0;
    while (off < len) {
        const std::uint32_t chunk = len - off;
        const int rc = SSL_write(ssl, static_cast<const char*>(data) + off,
                                 static_cast<int>(chunk));
        if (rc > 0) { off += static_cast<std::uint32_t>(rc); continue; }
        const int err = SSL_get_error(ssl, rc);
        if (err == SSL_ERROR_WANT_READ) {
            if (!sock_wait(fd, true, false, deadline)) return -1;
        } else if (err == SSL_ERROR_WANT_WRITE) {
            if (!sock_wait(fd, false, true, deadline)) return -1;
        } else {
            return -1;
        }
    }
    return static_cast<std::int64_t>(off);
}

bool tls_recv_all(const TlsSocket* s, void* data, std::uint32_t len) noexcept {
    assert(s != nullptr && "tls_recv_all: null s");
    assert(data != nullptr && "tls_recv_all: null data");
    if (!s->handshake_done || !s->ssl) return false;
    SSL* ssl = static_cast<SSL*>(s->ssl);
    const sock_t fd = static_cast<sock_t>(s->fd);
    const std::int64_t deadline = deadline_from(0, kDefaultRecvMs);

    std::uint32_t off = 0;
    while (off < len) {
        const int rc = SSL_read(ssl, static_cast<char*>(data) + off,
                                static_cast<int>(len - off));
        if (rc > 0) { off += static_cast<std::uint32_t>(rc); continue; }
        const int err = SSL_get_error(ssl, rc);
        if (err == SSL_ERROR_WANT_READ) {
            if (!sock_wait(fd, true, false, deadline)) return false;
        } else if (err == SSL_ERROR_WANT_WRITE) {
            if (!sock_wait(fd, false, true, deadline)) return false;
        } else {
            return false;
        }
    }
    return true;
}

std::int32_t tls_recv_some(const TlsSocket* s, void* data,
                           std::uint32_t cap) noexcept {
    assert(s != nullptr && "tls_recv_some: null s");
    assert(data != nullptr && "tls_recv_some: null data");
    if (!s->handshake_done || !s->ssl || cap == 0) return -1;
    SSL* ssl = static_cast<SSL*>(s->ssl);
    const sock_t fd = static_cast<sock_t>(s->fd);
    const std::int64_t deadline = deadline_from(0, kDefaultRecvMs);

    for (int spins = 0; spins < 256; ++spins) {
        const int rc = SSL_read(ssl, data, static_cast<int>(cap));
        if (rc > 0) return static_cast<std::int32_t>(rc);
        const int err = SSL_get_error(ssl, rc);
        if (err == SSL_ERROR_ZERO_RETURN) return 0;
        if (err == SSL_ERROR_WANT_READ) {
            if (!sock_wait(fd, true, false, deadline)) return -1;
        } else if (err == SSL_ERROR_WANT_WRITE) {
            if (!sock_wait(fd, false, true, deadline)) return -1;
        } else {
            return -1;
        }
    }
    return -1;
}

void tls_close(TlsSocket* s) noexcept {
    if (!s) return;
    if (s->ssl) {
        SSL* ssl = static_cast<SSL*>(s->ssl);
        // Best-effort shutdown; ignore errors (peer may already be gone).
        (void)SSL_shutdown(ssl);
        SSL_free(ssl);
    }
    if (s->ssl_ctx) SSL_CTX_free(static_cast<SSL_CTX*>(s->ssl_ctx));
    if (s->fd >= 0 && static_cast<sock_t>(s->fd) != kInvalidSock) {
        BOLT_TLS_CLOSESOCK(static_cast<sock_t>(s->fd));
    }
    std::memset(s, 0, sizeof(*s));
}

#endif  // BOLT_WITH_TLS

}  // namespace bolt::net
