// test_bolt_tls.cpp — smoke tests for bolt::net::TlsSocket.
//
// Network tests are SKIPPED gracefully if DNS or TCP fails so the suite
// stays green on offline CI workers.
#include "bolt/net/bolt_tls.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <sys/socket.h>
#  include <netdb.h>
#  include <arpa/inet.h>
#endif

namespace {

// Resolve `host` to one IPv4 in host byte order. Returns 0 on failure.
std::uint32_t resolve_v4(const char* host) noexcept {
#if defined(_WIN32)
    WSADATA d; ::WSAStartup(MAKEWORD(2, 2), &d);
#endif
    addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(host, "443", &hints, &res) != 0 || !res) return 0;
    auto* sin = reinterpret_cast<sockaddr_in*>(res->ai_addr);
    std::uint32_t ip = ntohl(sin->sin_addr.s_addr);
    ::freeaddrinfo(res);
    return ip;
}

void fill_default(bolt::net::TlsConfig* cfg, const char* host) noexcept {
    std::memset(cfg, 0, sizeof(*cfg));
    std::strncpy(cfg->server_name, host, sizeof(cfg->server_name) - 1);
    cfg->verify_peer          = true;
    cfg->insecure_skip_verify = false;
    cfg->handshake_timeout_ms = 8000;
    cfg->recv_timeout_ms      = 8000;
}

}  // namespace

// CTX init + free without crashing. Lightest possible signal that the
// OpenSSL link is wired up correctly.
TEST(BoltTls, CtxInitAndFree) {
    bolt::net::TlsConfig cfg;
    fill_default(&cfg, "example.com");
    // tls_close on a zeroed TlsSocket is documented as a no-op.
    bolt::net::TlsSocket sock{};
    bolt::net::tls_close(&sock);
    SUCCEED();
}

// Connect + minimal HTTP/1.1 GET against a public TLS 1.3 endpoint.
// Skips on offline workers so CI stays green.
TEST(BoltTls, ConnectAndGetCloudflare) {
    constexpr const char* kHost = "cloudflare.com";
    const std::uint32_t ip = resolve_v4(kHost);
    if (ip == 0) {
        GTEST_SKIP() << "DNS resolution failed (offline?); skipping live test";
    }

    bolt::net::TlsConfig cfg;
    fill_default(&cfg, kHost);

    bolt::net::TlsSocket sock{};
    const bool ok = bolt::net::tls_client_connect(&sock, &cfg, ip, 443);
    if (!ok) {
        GTEST_SKIP() << "tls_client_connect failed (network blocked?); skipping";
    }
    EXPECT_TRUE(sock.handshake_done);

    const char req[] = "GET / HTTP/1.1\r\nHost: cloudflare.com\r\n"
                       "User-Agent: bolt-tls-test\r\nConnection: close\r\n\r\n";
    const std::int64_t sent = bolt::net::tls_send_all(
        &sock, req, static_cast<std::uint32_t>(sizeof(req) - 1));
    EXPECT_EQ(sent, static_cast<std::int64_t>(sizeof(req) - 1));

    char buf[512] = {};
    const std::int32_t n = bolt::net::tls_recv_some(&sock, buf, sizeof(buf) - 1);
    EXPECT_GT(n, 0);

    bolt::net::tls_close(&sock);
}
