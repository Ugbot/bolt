#include "bolt/net/bolt_http_client.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using TestSocket = SOCKET;
using TestIoSize = int;
constexpr TestSocket kBadSocket = INVALID_SOCKET;
static void close_socket(TestSocket fd) { closesocket(fd); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using TestSocket = int;
using TestIoSize = std::size_t;
constexpr TestSocket kBadSocket = -1;
static void close_socket(TestSocket fd) { ::close(fd); }
#endif

namespace {
namespace bn = bolt::net;
using Pieces = std::vector<std::string>;

bool socket_startup() {
#ifdef _WIN32
    static const bool ok = [] { WSADATA wsa; return WSAStartup(MAKEWORD(2, 2), &wsa) == 0; }();
    return ok;
#else
    return true;
#endif
}

bool socket_send(TestSocket fd, const std::string& bytes) {
    for (size_t off = 0; off < bytes.size();) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const int n = static_cast<int>(::send(fd, bytes.data() + off,
            static_cast<TestIoSize>(bytes.size() - off), flags));
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

// One real connection per instance. All server waits have finite bounds, and
// request bytes are inspected only after joining its thread.
struct OneShotServer {
    TestSocket listener{kBadSocket};
    uint16_t port{0};
    std::atomic<bool> stopping{false};
    std::atomic<bool> accepted{false};
    std::thread worker;
    std::string request;

    bool start(Pieces pieces, unsigned stall_ms = 0) {
        if (!socket_startup()) return false;
        listener = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener == kBadSocket) return false;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listener, 1) != 0) return false;
#ifdef _WIN32
        int size = sizeof(address);
#else
        socklen_t size = sizeof(address);
#endif
        if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0) return false;
        port = ntohs(address.sin_port);
        worker = std::thread([this, pieces = std::move(pieces), stall_ms] { serve(pieces, stall_ms); });
        return true;
    }

    void serve(const Pieces& pieces, unsigned stall_ms) {
        TestSocket client = kBadSocket;
        for (unsigned i = 0; i < 20 && !stopping.load(); ++i) {
            fd_set reads;
            FD_ZERO(&reads);
            FD_SET(listener, &reads);
            timeval timeout{0, 100000};
#ifdef _WIN32
            const int count = 0;
#else
            const int count = listener + 1;
#endif
            if (::select(count, &reads, nullptr, nullptr, &timeout) > 0) {
                client = ::accept(listener, nullptr, nullptr);
                break;
            }
        }
        if (client == kBadSocket) return;
        accepted.store(true);
#ifdef SO_NOSIGPIPE
        const int one = 1;
        (void)::setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
#ifdef _WIN32
        DWORD timeout = 1000;
#else
        timeval timeout{1, 0};
#endif
        (void)::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        (void)::setsockopt(client, SOL_SOCKET, SO_SNDTIMEO,
            reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        char bytes[1024];
        for (unsigned i = 0; i < 32 && request.find("\r\n\r\n") == std::string::npos; ++i) {
            const int n = static_cast<int>(::recv(client, bytes, sizeof(bytes), 0));
            if (n <= 0) break;
            request.append(bytes, static_cast<size_t>(n));
        }
        const auto head_end = request.find("\r\n\r\n");
        const auto length_at = request.find("\r\nContent-Length: ");
        const auto body_size = length_at == std::string::npos ? 0u :
            std::strtoul(request.c_str() + length_at + 18, nullptr, 10);
        for (unsigned i = 0; i < 32 && head_end != std::string::npos &&
            body_size <= 32768 && request.size() < head_end + 4 + body_size; ++i) {
            const int n = static_cast<int>(::recv(client, bytes, sizeof(bytes), 0));
            if (n <= 0) break;
            request.append(bytes, static_cast<size_t>(n));
        }
        for (const auto& piece : pieces) {
            if (!socket_send(client, piece)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        for (unsigned i = 0; i < stall_ms && !stopping.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        close_socket(client);
    }

    void join() {
        if (worker.joinable()) worker.join();
    }
    ~OneShotServer() {
        stopping.store(true);
        join();
        if (listener != kBadSocket) close_socket(listener);
    }
};

void init_request(bn::HttpRequest* req, uint16_t port) {
    *req = {};
    std::memcpy(req->method, "GET", 4);
    std::snprintf(req->url, sizeof(req->url), "http://127.0.0.1:%u/path?x=1", port);
    req->timeout_ms = 1000;
}

TEST(BoltHttpClient, TruncatedContentLengthNeverPublishesBody) {
    for (const auto* suffix : {"", "abc"}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({std::string("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n") + suffix}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
        EXPECT_EQ(response.body, nullptr);
        EXPECT_EQ(response.body_len, 0u);
    }
}

TEST(BoltHttpClient, ChunkedReadsCrossReceiveBuffersAndConsumeTrailers) {
    const std::string body(20000, 'x');
    OneShotServer server;
    // The chunk exceeds the initial receive capacity even if TCP coalesces
    // every write; correctness does not depend on the 10 ms pacing above.
    ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        "4e20\r", "\n" + body.substr(0, 10000), body.substr(10000) + "\r\n0\r\n",
        "X-Checksum: abc\r", "\n\r\n"}));
    bn::HttpRequest req{};
    init_request(&req, server.port);
    bolt::Arena arena;
    bn::HttpResponse response{};
    ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
    EXPECT_EQ(response.body_len, body.size());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(response.body), response.body_len), body);
}

TEST(BoltHttpClient, ChunkSizeLineCrossesTheInitialReceiveCapacity) {
    OneShotServer server;
    const std::string chunk = "1;" + std::string(8180, 'a') + "\r\nx\r\n0\r\n\r\n";
    ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + chunk}));
    bn::HttpRequest req{};
    init_request(&req, server.port);
    bolt::Arena arena;
    bn::HttpResponse response{};
    ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
    ASSERT_EQ(response.body_len, 1u);
    EXPECT_EQ(response.body[0], static_cast<uint8_t>('x'));
}

TEST(BoltHttpClient, ExactContentLengthAcrossReadsAndHeadRepresentationLength) {
    const std::string body(20000, 'c');
    for (const bool head : {false, true}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nContent-Length: 20000\r\n\r\n" + (head ? "" : body)}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        if (head) std::memcpy(req.method, "HEAD", 5);
        req.max_response_body = head ? 1u : static_cast<uint32_t>(body.size());
        bolt::Arena arena;
        bn::HttpResponse response{};
        ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
        EXPECT_EQ(response.body_len, head ? 0u : body.size());
        if (!head) EXPECT_EQ(std::string(reinterpret_cast<const char*>(response.body), response.body_len), body);
    }
}

TEST(BoltHttpClient, EofDelimitedBodyConsumesEveryReceiveBuffer) {
    const std::string body(20000, 'e');
    OneShotServer server;
    ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\n\r\n" + body}));
    bn::HttpRequest req{};
    init_request(&req, server.port);
    bolt::Arena arena;
    bn::HttpResponse response{};
    ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
    EXPECT_EQ(response.body_len, body.size());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(response.body), response.body_len), body);
}

TEST(BoltHttpClient, IncompleteOrMalformedChunkFramingFails) {
    for (const auto* body : {"G\r\n", "1\r\naXX0\r\n\r\n", "0\r\n", "0\r\nX-End: y\r\n",
                             "10000000000000001\r\nx\r\n0\r\n\r\n"}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({std::string("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n") + body}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0) << body;
        EXPECT_EQ(response.body, nullptr);
        EXPECT_EQ(response.body_len, 0u);
    }
}

TEST(BoltHttpClient, InvalidOrOversizedHeadersRefuseWithoutTruncation) {
    const Pieces invalid{
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx",
        "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\nx",
        "HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n",
        "HTTP/1.1 200 OK\r\nX-Large: " + std::string(bn::kHttpMaxHeader, 'v') + "\r\n\r\n",
        "HTTP/1.1 200 OK\r\nX: partial"};
    for (const auto& bytes : invalid) {
        OneShotServer server;
        ASSERT_TRUE(server.start({bytes}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
        EXPECT_EQ(response.body, nullptr);
    }
}

TEST(BoltHttpClient, ResponseBudgetAppliesToAllFramingModes) {
    for (const auto* bytes : {"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabcd",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n",
            "HTTP/1.1 200 OK\r\n\r\nabcd"}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({bytes}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        req.max_response_body = 3;
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
        EXPECT_EQ(response.body, nullptr);
        EXPECT_EQ(response.body_len, 0u);
    }
}

TEST(BoltHttpClient, PlainDeadlineBoundsStalledHeadersAndBody) {
    for (const auto* prefix : {"", "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\na"}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({prefix}, 1000));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        req.timeout_ms = 60;
        bolt::Arena arena;
        bn::HttpResponse response{};
        const auto start = std::chrono::steady_clock::now();
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(750));
        EXPECT_EQ(response.body, nullptr);
    }
}

TEST(BoltHttpClient, PlainDeadlineIsNotRenewedByBodyProgress) {
    Pieces pieces{"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n"};
    for (unsigned i = 0; i < 20; ++i) pieces.emplace_back("x");
    OneShotServer server;
    ASSERT_TRUE(server.start(std::move(pieces)));
    bn::HttpRequest req{};
    init_request(&req, server.port);
    req.timeout_ms = 60;
    bolt::Arena arena;
    bn::HttpResponse response{};
    EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
    EXPECT_EQ(response.body, nullptr);
    EXPECT_EQ(response.body_len, 0u);
}

TEST(BoltHttpClient, NondefaultAuthorityAndFullHeaderValueReachWire) {
    const std::string token(2047, 't');
    OneShotServer server;
    ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Large: " + token + "\r\n\r\n"}));
    bn::HttpRequest req{};
    init_request(&req, server.port);
    ASSERT_TRUE(bn::http_request_add_header(&req, "X-Session", token.c_str()));
    bolt::Arena arena;
    bn::HttpResponse response{};
    ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
    ASSERT_NE(bn::http_header_get(response.headers, response.header_count, "X-Large"), nullptr);
    EXPECT_STREQ(bn::http_header_get(response.headers, response.header_count, "X-Large"), token.c_str());
    server.join();
    EXPECT_NE(server.request.find("\r\nHost: 127.0.0.1:" + std::to_string(server.port) + "\r\n"), std::string::npos);
    EXPECT_NE(server.request.find("\r\nX-Session: " + token + "\r\n"), std::string::npos);
    const std::string oversized(bn::kHttpMaxHeader, 'x');
    EXPECT_FALSE(bn::http_request_add_header(&req, "X-Overflow", oversized.c_str()));
    EXPECT_FALSE(bn::http_request_add_header(&req, "Bad\r\nName", "v"));
    EXPECT_FALSE(bn::http_request_add_header(&req, "X-Value", "v\r\nInjected: y"));
    EXPECT_EQ(req.header_count, 1u);
}

TEST(BoltHttpClient, AuthorityParsingPreservesPortsAndIpv6Brackets) {
    const std::pair<const char*, const char*> cases[]{
        {"http://example.test/path", "example.test"}, {"https://example.test:443/", "example.test"},
        {"http://example.test:9000/", "example.test:9000"},
        {"https://[::1]:9443/path", "[::1]:9443"}, {"http://[::1]:80/", "[::1]"}};
    char authority[256];
    for (const auto& item : cases) {
        ASSERT_TRUE(bn::http_url_authority(item.first, authority, sizeof(authority)));
        EXPECT_STREQ(authority, item.second);
    }
    for (const auto* url : {"http://host:1x/", "http://host:0/", "http://host:65536/",
                            "http://user@host/", "http://[::1/", "http://host/\r\nX: y"})
        EXPECT_FALSE(bn::http_url_authority(url, authority, sizeof(authority))) << url;
    EXPECT_FALSE(bn::http_url_authority("http://host:9000/", authority, 4));
}

TEST(BoltHttpClient, InvalidRequestBudgetsAndFieldsRefuseBeforeConnecting) {
    for (unsigned mode = 0; mode < 4; ++mode) {
        OneShotServer server;
        ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        if (mode == 0) req.header_count = bn::kHttpMaxHeaders + 1;
        if (mode == 1) req.body_len = 1; // a nonempty borrowed body needs a pointer
        if (mode == 2) req.timeout_ms = 60001;
        if (mode == 3) req.max_response_body = bn::kHttpMaxBody + 1;
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0);
        server.stopping.store(true);
        server.join();
        EXPECT_FALSE(server.accepted.load());
        EXPECT_EQ(response.status, 0);
        EXPECT_EQ(response.body, nullptr);
        EXPECT_EQ(response.body_len, 0u);
    }
}

TEST(BoltHttpClient, CallerFramingOverridesRefuseBeforeConnecting) {
    const std::pair<const char*, const char*> invalid[]{
        {"Host", "other"}, {"Connection", "close"}, {"Transfer-Encoding", "chunked"},
        {"Content-Length", "1"}, {"Content-Length", "-1"}, {"Content-Length", "0x"},
        {"Content-Length", "4294967296"}, {"Content-Length", "0"}};
    for (const auto& field : invalid) {
        OneShotServer server;
        ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        ASSERT_TRUE(bn::http_request_add_header(&req, field.first, field.second));
        if (std::strcmp(field.second, "0") == 0)
            ASSERT_TRUE(bn::http_request_add_header(&req, "content-length", "0"));
        bolt::Arena arena;
        bn::HttpResponse response{};
        EXPECT_LT(bn::http_send(&arena, &req, &response), 0) << field.first << ':' << field.second;
        server.stopping.store(true);
        server.join();
        EXPECT_FALSE(server.accepted.load());
        EXPECT_EQ(response.body, nullptr);
    }
}

TEST(BoltHttpClient, MatchingExplicitContentLengthIsSerializedExactlyOnce) {
    for (const bool nonempty : {false, true}) {
        OneShotServer server;
        ASSERT_TRUE(server.start({"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"}));
        bn::HttpRequest req{};
        init_request(&req, server.port);
        if (nonempty) {
            req.body = reinterpret_cast<const uint8_t*>("z");
            req.body_len = 1;
        }
        ASSERT_TRUE(bn::http_request_add_header(&req, "Content-Length", nonempty ? " 01 " : "0000"));
        bolt::Arena arena;
        bn::HttpResponse response{};
        ASSERT_EQ(bn::http_send(&arena, &req, &response), 0);
        server.join();
        const std::string expected = nonempty ? "\r\nContent-Length: 1\r\n" : "\r\nContent-Length: 0\r\n";
        const auto first = server.request.find(expected);
        EXPECT_NE(first, std::string::npos);
        EXPECT_EQ(server.request.rfind("\r\nContent-Length:"), first);
        if (nonempty) EXPECT_TRUE(server.request.ends_with("\r\n\r\nz"));
    }
}
} // namespace
