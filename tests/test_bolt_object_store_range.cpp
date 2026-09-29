// test_bolt_object_store_range.cpp — ranged GET on the object-store trait
// (G2ICE-235): filesystem pread, S3 `Range:` against a local HTTP server that
// honours it (206) and one that ignores it (200, whole object).

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "bolt/lakehouse/object_store.h"

using namespace bolt::lakehouse;

namespace {

std::vector<uint8_t> pattern(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>((i * 131u) ^ (i >> 7));
    return v;
}

TEST(ObjectStoreRange, FilesystemReadsExactRanges) {
    const auto root = std::filesystem::temp_directory_path() / "bolt_os_range";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    FilesystemObjectStore fs;
    ObjectStore os;
    ASSERT_TRUE(filesystem_object_store_init(&fs, root.string().c_str(), &os));
    const auto body = pattern(3u * 1024u * 1024u + 17u);
    ASSERT_EQ(os_put(&os, "a/b.bin", body.data(), body.size()), kOsOk);
    std::vector<uint8_t> buf(1u << 20);
    uint64_t got = 0;
    for (uint64_t off : {0ull, 1ull, 1048576ull, 2097152ull + 5ull}) {
        ASSERT_EQ(os_get_range(&os, "a/b.bin", off, buf.size(), buf.data(), &got), kOsOk);
        ASSERT_EQ(got, buf.size());
        ASSERT_EQ(std::memcmp(buf.data(), body.data() + off, got), 0) << off;
    }
    ASSERT_EQ(os_get_range(&os, "a/b.bin", 3u * 1024u * 1024u, buf.size(), buf.data(), &got), kOsOk);
    EXPECT_EQ(got, 17u) << "short only at the end of the object";
    EXPECT_EQ(std::memcmp(buf.data(), body.data() + 3u * 1024u * 1024u, 17), 0);
    ASSERT_EQ(os_get_range(&os, "a/b.bin", body.size() + 10u, 8, buf.data(), &got), kOsOk);
    EXPECT_EQ(got, 0u);
    EXPECT_EQ(os_get_range(&os, "missing", 0, 8, buf.data(), &got), kOsNotFound);

    ObjectStoreVT no_range = *os.vt;
    no_range.get_range = nullptr;
    ObjectStore os2{&no_range, os.impl};
    EXPECT_EQ(os_get_range(&os2, "a/b.bin", 0, 8, buf.data(), &got), kOsNotImplemented);
}

// One-connection-per-request HTTP server over a fixed body.
struct RangeServer {
    int fd = -1;
    uint16_t port = 0;
    bool honour = true;
    std::vector<uint8_t> body;
    std::atomic<int> requests{0};
    std::string last_range;
    std::thread th;
    std::atomic<bool> stop{false};

    void start() {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        ASSERT_GE(fd, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ASSERT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)), 0);
        socklen_t al = sizeof(a);
        ASSERT_EQ(::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &al), 0);
        port = ntohs(a.sin_port);
        ASSERT_EQ(::listen(fd, 8), 0);
        th = std::thread([this] { loop(); });
    }
    void loop() {
        while (!stop.load()) {
            const int c = ::accept(fd, nullptr, nullptr);
            if (c < 0) return;
            std::string req;
            char b[4096];
            while (req.find("\r\n\r\n") == std::string::npos) {
                const ssize_t n = ::recv(c, b, sizeof(b), 0);
                if (n <= 0) break;
                req.append(b, static_cast<size_t>(n));
            }
            requests.fetch_add(1);
            uint64_t lo = 0, hi = body.size() - 1u;
            const size_t r = req.find("Range: bytes=");
            last_range.clear();
            if (r != std::string::npos) {
                last_range = req.substr(r + 13, req.find("\r\n", r) - r - 13);
                std::sscanf(last_range.c_str(), "%llu-%llu",
                            reinterpret_cast<unsigned long long*>(&lo),
                            reinterpret_cast<unsigned long long*>(&hi));
                if (hi >= body.size()) hi = body.size() - 1u;
            }
            const bool partial = honour && r != std::string::npos;
            if (!partial) { lo = 0; hi = body.size() - 1u; }
            char head[256];
            const int hn = std::snprintf(head, sizeof(head),
                "HTTP/1.1 %s\r\nContent-Length: %llu\r\nConnection: close\r\n\r\n",
                partial ? "206 Partial Content" : "200 OK",
                static_cast<unsigned long long>(hi - lo + 1u));
            (void)::send(c, head, static_cast<size_t>(hn), 0);
            (void)::send(c, body.data() + lo, hi - lo + 1u, 0);
            ::close(c);
        }
    }
    void finish() {
        stop.store(true);
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        if (th.joinable()) th.join();
    }
};

void s3_range_case(bool honour) {
    RangeServer srv;
    srv.honour = honour;
    srv.body = pattern(200000);
    srv.start();
    S3ObjectStore s3;
    ObjectStore os;
    ASSERT_TRUE(s3_object_store_init(&s3, "bkt", "us-east-1", "ak", "sk", &os));
    std::snprintf(s3.endpoint, sizeof(s3.endpoint), "http://127.0.0.1:%u", srv.port);
    s3.path_style = true;
    std::vector<uint8_t> buf(65536);
    uint64_t got = 0;
    ASSERT_EQ(os_get_range(&os, "native/x.mblt", 70000, buf.size(), buf.data(), &got), kOsOk);
    ASSERT_EQ(got, buf.size());
    EXPECT_EQ(std::memcmp(buf.data(), srv.body.data() + 70000, got), 0);
    EXPECT_EQ(srv.last_range, "70000-135535");
    ASSERT_EQ(os_get_range(&os, "native/x.mblt", 196608, buf.size(), buf.data(), &got), kOsOk);
    EXPECT_EQ(got, 200000u - 196608u);
    EXPECT_EQ(std::memcmp(buf.data(), srv.body.data() + 196608, got), 0);
    EXPECT_EQ(srv.requests.load(), 2);
    srv.finish();
}

TEST(ObjectStoreRange, S3SendsRangeAndReads206) { s3_range_case(true); }
TEST(ObjectStoreRange, S3CutsRangeWhenServerIgnoresIt) { s3_range_case(false); }

}  // namespace
