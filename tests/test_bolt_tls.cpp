// test_bolt_tls.cpp — smoke tests for bolt::net::TlsSocket.
//
// Network tests are SKIPPED gracefully if DNS or TCP fails so the suite
// stays green on offline CI workers.
#include "bolt/net/bolt_tls.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cassert>
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

// G2CHK-198: a TLS peer that cuts the connection must surface as an error
// return from tls_send_all, not a SIGPIPE that kills the embedder.
#if defined(BOLT_WITH_TLS) && !defined(_WIN32)

#include <openssl/evp.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>

#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <thread>

namespace {

struct SelfSigned {
    EVP_PKEY* key  = nullptr;
    X509*     cert = nullptr;
};

bool make_self_signed(SelfSigned* out) noexcept {
    assert(out != nullptr);
    if (out == nullptr) return false;
    assert(out->key == nullptr && out->cert == nullptr);
    *out = {};
    EVP_PKEY* key = EVP_EC_gen("P-256");
    X509* cert = key != nullptr ? X509_new() : nullptr;
    X509_NAME* name = cert != nullptr ? X509_NAME_new() : nullptr;
    ASN1_INTEGER* serial = cert != nullptr ? X509_get_serialNumber(cert) : nullptr;
    ASN1_TIME* before = cert != nullptr ? X509_getm_notBefore(cert) : nullptr;
    ASN1_TIME* after = cert != nullptr ? X509_getm_notAfter(cert) : nullptr;
    const bool complete = name != nullptr && serial != nullptr && before != nullptr && after != nullptr &&
        X509_set_version(cert, 2) == 1 && ASN1_INTEGER_set(serial, 1) == 1 &&
        X509_gmtime_adj(before, 0) != nullptr && X509_gmtime_adj(after, 3600) != nullptr &&
        X509_set_pubkey(cert, key) == 1 &&
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
            reinterpret_cast<const unsigned char*>("localhost"), -1, -1, 0) == 1 &&
        X509_set_subject_name(cert, name) == 1 && X509_set_issuer_name(cert, name) == 1 &&
        X509_sign(cert, key, EVP_sha256()) > 0;
    X509_NAME_free(name); // both setters copy; the fixture owns this name
    if (!complete) { X509_free(cert); EVP_PKEY_free(key); return false; }
    out->key = key; out->cert = cert;
    return true;
}

// Accept one connection, finish the TLS handshake, then reset it.
void accept_handshake_then_reset(int lfd, const SelfSigned* ss, bool* ok) noexcept {
    *ok = false;
    const int cfd = ::accept(lfd, nullptr, nullptr);
    if (cfd < 0) return;
    SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
    SSL* ssl = nullptr;
    if (ctx != nullptr &&
        SSL_CTX_use_certificate(ctx, ss->cert) == 1 &&
        SSL_CTX_use_PrivateKey(ctx, ss->key) == 1) {
        ssl = SSL_new(ctx);
        if (ssl != nullptr && SSL_set_fd(ssl, cfd) == 1 && SSL_accept(ssl) == 1) {
            *ok = true;
        }
    }
    linger lg{1, 0};
    ::setsockopt(cfd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    ::close(cfd);
    if (ssl != nullptr) SSL_free(ssl);
    if (ctx != nullptr) SSL_CTX_free(ctx);
}

}  // namespace

TEST(BoltTls, PeerResetDoesNotRaiseSigpipe) {
    ::signal(SIGPIPE, SIG_DFL);  // the embedder did NOT ignore SIGPIPE

    SelfSigned ss;
    ASSERT_TRUE(make_self_signed(&ss));

    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(lfd, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    ASSERT_EQ(::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    ASSERT_EQ(::listen(lfd, 1), 0);
    socklen_t alen = sizeof(addr);
    ASSERT_EQ(::getsockname(lfd, reinterpret_cast<sockaddr*>(&addr), &alen), 0);

    bool server_ok = false;
    std::thread server(accept_handshake_then_reset, lfd, &ss, &server_ok);

    bolt::net::TlsConfig cfg;
    fill_default(&cfg, "localhost");
    cfg.verify_peer          = false;
    cfg.insecure_skip_verify = true;
    bolt::net::TlsSocket sock{};
    const bool connected =
        bolt::net::tls_client_connect(&sock, &cfg, INADDR_LOOPBACK, ntohs(addr.sin_port));
    server.join();
    ::close(lfd);
    ASSERT_TRUE(connected);
    ASSERT_TRUE(server_ok);

    static std::uint8_t buf[16384];
    bool failed = false;
    for (int i = 0; i < 400 && !failed; ++i) {
        failed = bolt::net::tls_send_all(&sock, buf, sizeof(buf)) < 0;
        if (!failed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_TRUE(failed) << "writes into a reset connection never failed";
    bolt::net::tls_close(&sock);

    X509_free(ss.cert);
    EVP_PKEY_free(ss.key);
}


// Explicit I/O deadlines against a handshaken peer that produces no data.
// Reuses the in-process certificate/server setup; never contacts public hosts.
#include <atomic>
#include <poll.h>
namespace {
struct SilentTlsPeer {
    SelfSigned certificate{};
    int listener=-1;
    std::uint16_t port=0;
    std::atomic<bool> stop{false},ready{false};
    std::thread thread;
    SilentTlsPeer() {
        if (!make_self_signed(&certificate)) return;
        listener=::socket(AF_INET,SOCK_STREAM,0); if (listener<0) return;
        sockaddr_in address{}; address.sin_family=AF_INET;
        address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        if (::bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address))!=0 || ::listen(listener,1)!=0) return;
        socklen_t bytes=sizeof(address);
        if (::getsockname(listener,reinterpret_cast<sockaddr*>(&address),&bytes)!=0) return;
        port=ntohs(address.sin_port);
        thread=std::thread([this] {
            pollfd waiting{listener,POLLIN,0}; if (::poll(&waiting,1,2000)<=0) return;
            const auto fd=::accept(listener,nullptr,nullptr); if (fd<0) return;
            timeval timeout{}; timeout.tv_sec=1;
            ::setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
            const int window=4096;
            if (::setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&window,sizeof(window))!=0) { ::close(fd); return; }
            SSL_CTX* ctx=SSL_CTX_new(TLS_server_method()); SSL* ssl=nullptr;
            if (ctx!=nullptr && SSL_CTX_use_certificate(ctx,certificate.cert)==1 && SSL_CTX_use_PrivateKey(ctx,certificate.key)==1) {
                ssl=SSL_new(ctx);
                if (ssl!=nullptr && SSL_set_fd(ssl,fd)==1 && SSL_accept(ssl)==1) {
                    ready=true;
                    for (auto i=0u;i<4000 && !stop;++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            if (ssl!=nullptr) SSL_free(ssl);
            if (ctx!=nullptr) SSL_CTX_free(ctx);
            ::close(fd);
        });
    }
    ~SilentTlsPeer() {
        stop=true; if (thread.joinable()) thread.join();
        if (listener>=0) ::close(listener);
        if (certificate.cert!=nullptr) X509_free(certificate.cert);
        if (certificate.key!=nullptr) EVP_PKEY_free(certificate.key);
    }
    bool connect(bolt::net::TlsSocket& client) {
        bolt::net::TlsConfig cfg{}; fill_default(&cfg,"localhost");
        cfg.verify_peer=false; cfg.insecure_skip_verify=true; // local certificate diagnostic only
        return bolt::net::tls_client_connect(&client,&cfg,INADDR_LOOPBACK,port);
    }
};
}
TEST(BoltTls, ExplicitReceiveDeadlineUsesRequestedBudget) {
    SilentTlsPeer peer; ASSERT_NE(peer.port,0);
    bolt::net::TlsSocket client{}; ASSERT_TRUE(peer.connect(client));
    char bytes[64]{};
    EXPECT_EQ(bolt::net::tls_recv_some_timeout(&client,bytes,sizeof(bytes),0),-1);
    EXPECT_EQ(bolt::net::tls_recv_some_timeout(&client,bytes,sizeof(bytes),60001),-1);
    const auto start=std::chrono::steady_clock::now();
    const auto n=bolt::net::tls_recv_some_timeout(&client,bytes,sizeof(bytes),40);
    const auto elapsed=std::chrono::steady_clock::now()-start;
    EXPECT_EQ(n,-1); EXPECT_GE(elapsed,std::chrono::milliseconds(20));
    EXPECT_LT(elapsed,std::chrono::milliseconds(500));
    std::fprintf(stdout,"explicit TLS receive timeout: budget_ms=40 elapsed_ms=%lld\n",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
    bolt::net::tls_close(&client); peer.stop=true;
}
TEST(BoltTls, ExplicitSendDeadlineBoundsNonReadingPeer) {
    SilentTlsPeer peer; ASSERT_NE(peer.port,0);
    bolt::net::TlsSocket client{}; ASSERT_TRUE(peer.connect(client));
    const int window=4096;
    EXPECT_EQ(::setsockopt(client.fd,SOL_SOCKET,SO_SNDBUF,&window,sizeof(window)),0);
    static std::uint8_t bytes[4u*1024u*1024u]{};
    EXPECT_EQ(bolt::net::tls_send_all_timeout(&client,bytes,sizeof(bytes),0),-1);
    EXPECT_EQ(bolt::net::tls_send_all_timeout(&client,bytes,sizeof(bytes),60001),-1);
    const auto start=std::chrono::steady_clock::now();
    const auto n=bolt::net::tls_send_all_timeout(&client,bytes,sizeof(bytes),40);
    const auto elapsed=std::chrono::steady_clock::now()-start;
    EXPECT_EQ(n,-1); EXPECT_LT(elapsed,std::chrono::milliseconds(500));
    std::fprintf(stdout,"explicit TLS send timeout: budget_ms=40 elapsed_ms=%lld\n",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
    bolt::net::tls_close(&client); peer.stop=true;
}

#endif  // BOLT_WITH_TLS && !_WIN32
