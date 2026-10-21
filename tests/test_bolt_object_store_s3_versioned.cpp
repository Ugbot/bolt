#include "bolt/lakehouse/object_store/range_cache.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/net/bolt_http_client.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using TestSocket = SOCKET;
using TestIoSize = int;
constexpr TestSocket kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using TestSocket = int;
using TestIoSize = size_t;
constexpr TestSocket kInvalidSocket = -1;
#endif

namespace {
using namespace bolt::lakehouse;
constexpr uint32_t kExtent = 257, kOffset = 17, kLength = 31;
constexpr const char* kTag = "\"opaque-v1\"";
uint8_t value(uint32_t offset) { return static_cast<uint8_t>((offset * 131u) ^ (offset >> 3u)); }
void close_socket(TestSocket socket) {
#ifdef _WIN32
    (void)::closesocket(socket);
#else
    (void)::close(socket);
#endif
}
void shutdown_socket(TestSocket socket) {
#ifdef _WIN32
    (void)::shutdown(socket, SD_BOTH);
#else
    (void)::shutdown(socket, SHUT_RDWR);
#endif
}
bool send_bytes(TestSocket socket, const std::string& bytes) {
    size_t off = 0;
    for (uint32_t part = 0; off < bytes.size() && part < 64; ++part) {
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const auto want = static_cast<TestIoSize>(std::min<size_t>(bytes.size() - off, 65536u));
        const int sent = static_cast<int>(::send(socket, bytes.data() + off, want, flags));
        if (sent <= 0) return false;
        off += static_cast<size_t>(sent);
    }
    return off == bytes.size();
}
std::string header(const std::string& request, const char* name) {
    const std::string find = std::string("\r\n") + name + ": ";
    const size_t start = request.find(find);
    if (start == std::string::npos) return {};
    const size_t value_start = start + find.size();
    const size_t end = request.find("\r\n", value_start);
    return end == std::string::npos ? std::string{} : request.substr(value_start, end - value_start);
}
enum class Reply { Valid, MissingEtag, DuplicateEtag, WeakEtag, ChangedEtag, OversizeEtag,
    MissingRange, DuplicateRange, WrongRange, OverflowRange, WrongTotal, IgnoredRange,
    Precondition, NotFound, Truncated, ShortBody, MissingLength, DuplicateLength, OverflowLength };

class VersionServer {
public:
    bool head = false, valid = true;
    Reply reply = Reply::Valid;
    uint16_t port = 0;
    uint32_t requests = 0;
    std::string token = std::string(640, 't');
    ~VersionServer() { stop(); }
    bool start() {
#ifdef _WIN32
        WSADATA data{};
        if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) return false;
        wsa_ = true;
#endif
        listener_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener_ == kInvalidSocket) return false;
        sockaddr_in address{}; address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = 0;
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listener_, 1) != 0) return false;
#ifdef _WIN32
        int size = sizeof(address);
#else
        socklen_t size = sizeof(address);
#endif
        if (::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size) != 0) return false;
        port = ntohs(address.sin_port);
        worker_ = std::thread([this] { serve(); });
        return true;
    }
    void stop() {
        if (listener_ != kInvalidSocket) { shutdown_socket(listener_); close_socket(listener_); }
        if (worker_.joinable()) worker_.join();
        listener_ = kInvalidSocket;
#ifdef _WIN32
        if (wsa_) { (void)::WSACleanup(); wsa_ = false; }
#endif
    }
    bool store(S3ObjectStore* s3, ObjectStore* out) {
        if (!s3_object_store_init(s3, "bucket", "us-east-1", "access", "secret", out)) return false;
        std::snprintf(s3->endpoint, sizeof(s3->endpoint), "http://127.0.0.1:%u/base", port);
        std::memcpy(s3->session_token, token.c_str(), token.size() + 1u);
        s3->path_style = true;
        return true;
    }
private:
    TestSocket listener_ = kInvalidSocket;
    std::thread worker_;
#ifdef _WIN32
    bool wsa_ = false;
#endif
    bool validate(const std::string& request) {
        const std::string method = head ? "HEAD" : "GET";
        if (!request.starts_with(method + " /base/bucket/key%20name.bin HTTP/1.1\r\n")) return false;
        char authority[64]{};
        std::snprintf(authority, sizeof(authority), "127.0.0.1:%u", port);
        if (header(request, "Host") != authority || header(request, "X-Amz-Security-Token") != token) return false;
        if (!head && (header(request, "Range") != "bytes=17-47" || header(request, "If-Match") != kTag)) return false;
        const std::string date = header(request, "X-Amz-Date");
        char empty[65]{}; bolt::crypto::sha256_empty_hex(empty);
        bolt::crypto::SigV4Request sign{};
        sign.method = method.c_str(); sign.host = authority; sign.path = "/base/bucket/key%20name.bin";
        sign.query_string = ""; sign.service = "s3"; sign.region = "us-east-1";
        sign.amz_date = date.c_str(); sign.payload_sha256 = empty; sign.access_key = "access";
        sign.secret_key = "secret"; sign.session_token = token.c_str(); sign.if_match = head ? nullptr : kTag;
        char auth[bolt::crypto::kSigV4MaxAuth]{}, date_out[24]{};
        bolt::crypto::SigV4Result result{auth, sizeof(auth), date_out, sizeof(date_out)};
        if (!bolt::crypto::sigv4_sign(&sign, &result) || header(request, "Authorization") != auth) return false;
        return head || header(request, "Authorization").find("SignedHeaders=host;if-match;x-amz-content-sha256;x-amz-date;x-amz-security-token,") != std::string::npos;
    }
    std::string response() {
        if (reply == Reply::Precondition) return "HTTP/1.1 412 Precondition Failed\r\nContent-Length: 0\r\n\r\n";
        if (reply == Reply::NotFound) return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        std::string out = head || reply == Reply::IgnoredRange ? "HTTP/1.1 200 OK\r\n" : "HTTP/1.1 206 Partial Content\r\n";
        if (reply != Reply::MissingEtag) {
            const std::string tag = reply == Reply::WeakEtag ? "W/\"opaque-v1\"" :
                reply == Reply::ChangedEtag ? "\"opaque-v2\"" : reply == Reply::OversizeEtag ? "\"" + std::string(256, 'e') + "\"" : kTag;
            out += "ETag: " + tag + "\r\n";
            if (reply == Reply::DuplicateEtag) out += "eTaG: " + tag + "\r\n";
        }
        if (!head && reply != Reply::MissingRange && reply != Reply::IgnoredRange) {
            const std::string range = reply == Reply::WrongRange ? "bytes 18-48/257" :
                reply == Reply::OverflowRange ? "bytes 17-18446744073709551616/257" :
                reply == Reply::WrongTotal ? "bytes 17-47/258" : "bytes 17-47/257";
            out += "Content-Range: " + range + "\r\n";
            if (reply == Reply::DuplicateRange) out += "Content-Range: " + range + "\r\n";
        }
        uint32_t size = head || reply == Reply::IgnoredRange ? kExtent : kLength;
        if (reply == Reply::ShortBody) --size;
        if (reply != Reply::MissingLength) out += "Content-Length: " +
            (reply == Reply::OverflowLength ? std::string("18446744073709551616") : std::to_string(size)) + "\r\n";
        if (reply == Reply::DuplicateLength) out += "Content-Length: " + std::to_string(size) + "\r\n";
        out += "Connection: close\r\n\r\n";
        if (!head) {
            if (reply == Reply::Truncated) --size;
            for (uint32_t i = 0; i < size; ++i) out += static_cast<char>(value((reply == Reply::IgnoredRange ? 0 : kOffset) + i));
        }
        return out;
    }
    void serve() {
        const TestSocket client = ::accept(listener_, nullptr, nullptr);
        if (client == kInvalidSocket) return;
#ifdef _WIN32
        const DWORD timeout = 3000;
        valid = ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout)) == 0;
#else
        const timeval timeout{3, 0};
        valid = ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0;
#endif
        std::string request;
        for (uint32_t attempt = 0; valid && attempt < 16384 && request.size() < 16384; ++attempt) {
            char bytes[1024]{};
            const int got = static_cast<int>(::recv(client, bytes, static_cast<TestIoSize>(sizeof(bytes)), 0));
            if (got <= 0) { valid = false; break; }
            request.append(bytes, static_cast<size_t>(got));
            if (request.find("\r\n\r\n") != std::string::npos) break;
        }
        ++requests;
        valid = valid && validate(request);
        if (valid) valid = send_bytes(client, response());
        shutdown_socket(client); close_socket(client);
    }
};

int head_request(Reply reply, ObjectVersion* version) {
    VersionServer server; server.head = true; server.reply = reply;
    if (!server.start()) { ADD_FAILURE() << "fixture start"; return kOsIoError; }
    S3ObjectStore s3{}; ObjectStore store{};
    if (!server.store(&s3, &store)) return kOsBadArg;
    bolt::Arena scratch{{kObjectRangePageBytes, kObjectRangePageBytes, 64, false}};
    scratch.set_growth_enabled(false);
    const int rc = os_head_versioned(&store, "key name.bin", &scratch, version);
    server.stop(); EXPECT_TRUE(server.valid); EXPECT_EQ(server.requests, 1u);
    return rc;
}
int range_request(Reply reply, std::array<uint8_t, kLength>* output, uint32_t* got) {
    VersionServer server; server.reply = reply;
    if (!server.start()) { ADD_FAILURE() << "fixture start"; return kOsIoError; }
    S3ObjectStore s3{}; ObjectStore store{};
    if (!server.store(&s3, &store)) return kOsBadArg;
    bolt::Arena scratch{{kObjectRangePageBytes, kObjectRangePageBytes, 64, false}};
    scratch.set_growth_enabled(false);
    ObjectVersion expected{}; expected.size = kExtent; std::strcpy(expected.etag, kTag);
    const int rc = os_get_range_if_match(&store, "key name.bin", &expected, kOffset, kLength,
                                        &scratch, output->data(), got);
    server.stop(); EXPECT_TRUE(server.valid); EXPECT_EQ(server.requests, 1u);
    return rc;
}

TEST(S3VersionedRange, HeadCopiesExactStrongOpaqueValidatorAndExtent) {
    ObjectVersion version{};
    ASSERT_EQ(head_request(Reply::Valid, &version), kOsOk);
    EXPECT_EQ(version.size, kExtent); EXPECT_STREQ(version.etag, kTag);
}
TEST(S3VersionedRange, HeadRefusesMissingDuplicateWeakOversizeAndUnboundedExtentHeaders) {
    for (const Reply reply : {Reply::MissingEtag, Reply::DuplicateEtag, Reply::WeakEtag,
                              Reply::OversizeEtag, Reply::MissingLength, Reply::DuplicateLength, Reply::OverflowLength}) {
        SCOPED_TRACE(static_cast<int>(reply));
        ObjectVersion version{}; version.size = 999;
        EXPECT_EQ(head_request(reply, &version), kOsIoError);
        EXPECT_EQ(version.size, 0u); EXPECT_STREQ(version.etag, "");
    }
    ObjectVersion absent{};
    EXPECT_EQ(head_request(Reply::NotFound, &absent), kOsNotFound);
}
TEST(S3VersionedRange, SignedIfMatchRangePublishesOnlyExactVerified206) {
    std::array<uint8_t, kLength> bytes{}; uint32_t got = 0;
    ASSERT_EQ(range_request(Reply::Valid, &bytes, &got), kOsOk); ASSERT_EQ(got, kLength);
    for (uint32_t i = 0; i < kLength; ++i) EXPECT_EQ(bytes[i], value(kOffset + i));
}
TEST(S3VersionedRange, MalformedIgnoredShortAndTruncatedResponsesNeverPublishBytes) {
    for (const Reply reply : {Reply::MissingEtag, Reply::DuplicateEtag, Reply::WeakEtag,
        Reply::OversizeEtag, Reply::MissingRange, Reply::DuplicateRange, Reply::WrongRange,
        Reply::OverflowRange, Reply::WrongTotal, Reply::IgnoredRange, Reply::Truncated, Reply::ShortBody}) {
        SCOPED_TRACE(static_cast<int>(reply));
        std::array<uint8_t, kLength> bytes{}; bytes.fill(0xa5); uint32_t got = 999;
        EXPECT_EQ(range_request(reply, &bytes, &got), kOsIoError); EXPECT_EQ(got, 0u);
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0xa5; }));
    }
}
TEST(S3VersionedRange, PreconditionAndChangedEtagAreExplicitWithoutRetry) {
    for (const Reply reply : {Reply::Precondition, Reply::ChangedEtag, Reply::NotFound}) {
        std::array<uint8_t, kLength> bytes{}; uint32_t got = 999;
        EXPECT_EQ(range_request(reply, &bytes, &got), reply == Reply::NotFound ? kOsNotFound : kOsPreconditionFailed);
        EXPECT_EQ(got, 0u);
    }
}
TEST(S3VersionedRange, AdmissionRejectsInvalidPinnedDescriptorAndRangeWithoutConnecting) {
    S3ObjectStore s3{}; ObjectStore store{};
    ASSERT_TRUE(s3_object_store_init(&s3, "bucket", "us-east-1", "access", "secret", &store));
    std::strcpy(s3.endpoint, "http://127.0.0.1:1"); s3.path_style = true;
    bolt::Arena scratch; ObjectVersion version{}; version.size = kExtent;
    uint8_t byte = 0; uint32_t got = 9;
    EXPECT_EQ(os_get_range_if_match(&store, "key", &version, 0, 1, &scratch, &byte, &got), kOsBadArg);
    EXPECT_EQ(got, 0u);
    std::strcpy(version.etag, kTag);
    EXPECT_EQ(os_get_range_if_match(&store, "key", &version, UINT64_MAX, 1, &scratch, &byte, &got), kOsBadArg);
    EXPECT_EQ(os_get_range_if_match(&store, "key", &version, 0, kObjectRangePageBytes + 1u, &scratch, &byte, &got), kOsBadArg);
}
}  // namespace

namespace {
TEST(S3VersionedRange, WrapperAndProviderAliasesRefuseBeforeConnectingAndPreserveInputs) {
    S3ObjectStore s3{}; ObjectStore store{};
    ASSERT_TRUE(s3_object_store_init(&s3, "bucket", "us-east-1", "access", "secret", &store));
    std::strcpy(s3.endpoint, "http://127.0.0.1:1"); s3.path_style = true;
    bolt::Arena scratch;
    ObjectVersion expected{}; expected.size = kExtent; std::strcpy(expected.etag, kTag);
    const auto saved = expected; uint8_t byte = 0; uint32_t got = 75;
    EXPECT_EQ(os_get_range_if_match(&store, "key", &expected, 0, 1, &scratch,
        reinterpret_cast<uint8_t*>(&expected), &got), kOsBadArg);
    EXPECT_EQ(store.vt->get_range_if_match(store.impl, "key", &expected, 0, 1, &scratch,
        &byte, reinterpret_cast<uint32_t*>(&expected.size)), kOsBadArg);
    const auto saved_s3 = s3;
    EXPECT_EQ(store.vt->get_range_if_match(store.impl, "key", &expected, 0, 1, &scratch,
        reinterpret_cast<uint8_t*>(&s3), &got), kOsBadArg);
    alignas(ObjectVersion) std::array<char, sizeof(ObjectVersion)> key{}; key[0] = 'k';
    const auto saved_key = key;
    EXPECT_EQ(store.vt->head_versioned(store.impl, key.data(), &scratch,
        reinterpret_cast<ObjectVersion*>(key.data())), kOsBadArg);
    EXPECT_EQ(std::memcmp(&s3, &saved_s3, sizeof(s3)), 0);
    EXPECT_EQ(std::memcmp(&expected, &saved, sizeof(expected)), 0);
    EXPECT_EQ(key, saved_key); EXPECT_EQ(got, 75u);
}
TEST(S3VersionedRange, DirectProviderRefusesResetWorkspaceOutputAndOverflowingAddress) {
    S3ObjectStore s3{}; ObjectStore store{};
    ASSERT_TRUE(s3_object_store_init(&s3, "bucket", "us-east-1", "access", "secret", &store));
    std::strcpy(s3.endpoint, "http://127.0.0.1:1"); s3.path_style = true;
    bolt::Arena scratch;
    auto* got = scratch.allocate_array<uint32_t>(1); ASSERT_NE(got, nullptr); *got = 61;
    ObjectVersion expected{}; expected.size = kExtent; std::strcpy(expected.etag, kTag);
    uint8_t byte = 0;
    EXPECT_EQ(store.vt->get_range_if_match(store.impl, "key", &expected, 0, 1, &scratch, &byte, got), kOsBadArg);
    const auto* bad = reinterpret_cast<const ObjectVersion*>(UINTPTR_MAX - 3u);
    uint32_t separate = 47;
    EXPECT_EQ(os_get_range_if_match(&store, "key", bad, 0, 1, &scratch, &byte, &separate), kOsBadArg);
    EXPECT_EQ(*got, 61u); EXPECT_EQ(separate, 47u);
}
}  // namespace
