#include "bolt/lakehouse/object_store/s3_upload.h"
#include "bolt/crypto/sigv4.h"
#include "bolt/net/bolt_http_client.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using UploadSocket = SOCKET;
using UploadIoSize = int;
constexpr UploadSocket kBadSocket = INVALID_SOCKET;
using UploadSockLen = int;
void close_upload_socket(UploadSocket fd) { ::closesocket(fd); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using UploadSocket = int;
using UploadIoSize = size_t;
constexpr UploadSocket kBadSocket = -1;
using UploadSockLen = socklen_t;
void close_upload_socket(UploadSocket fd) { ::close(fd); }
#endif

namespace {
using namespace bolt::lakehouse;
constexpr uint64_t kUploadBytes = 17u * 1024u * 1024u + 31u;
constexpr const char* kUploadKey = "native/a b+c.mblt";
constexpr const char* kUploadPath = "/base/bucket/native/a%20b%2Bc.mblt";

uint8_t pattern_byte(uint64_t offset) {
    return static_cast<uint8_t>((offset * 131u) ^ (offset >> 7u) ^ (offset >> 19u));
}

bool socket_send(UploadSocket fd, const std::string& text) {
    size_t offset = 0;
    while (offset < text.size()) {
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const int sent = static_cast<int>(::send(fd, text.data() + offset,
            static_cast<UploadIoSize>(std::min<size_t>(text.size() - offset, 65536u)), flags));
        if (sent <= 0) return false;
        offset += static_cast<size_t>(sent);
    }
    return true;
}

std::string request_header(const std::string& request, const char* name) {
    const std::string needle = std::string("\r\n") + name + ": ";
    const size_t start = request.find(needle);
    if (start == std::string::npos) return {};
    const size_t value = start + needle.size();
    return request.substr(value, request.find("\r\n", value) - value);
}

enum class UploadMode { Success, PartError, AbortError, MissingEtag, OversizeEtag,
    Error200, Truncated, Dropped, Conditional412, Conflict409, MalformedInit,
    WrongKey, MalformedComplete, ChunkedComplete, NumericId, InitOversize,
    InitDuplicate, InitBadEntity, InitBadDeclaration, SmallPutDropped, SmallPutTruncated,
    SmallPutServerError, SmallPutEtag };

struct UploadStats { unsigned parts = 0; uint64_t bytes = 0; };

class UploadServer {
public:
    UploadMode mode = UploadMode::Success;
    uint64_t total = kUploadBytes;
    uint32_t part_bytes = kS3DefaultPartBytes;
    unsigned creates = 0, completes = 0, aborts = 0, puts = 0;
    unsigned conditional_completes = 0;
    unsigned conditional_puts = 0;
    std::string accepted_put_body;
    std::array<UploadStats, 8> uploads{};
    bool valid = true;
    uint16_t port = 0;
    std::string token = std::string(768, 't');

    ~UploadServer() { stop(); }
    bool start() {
#ifdef _WIN32
        WSADATA data{};
        if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) return false;
#endif
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ == kBadSocket) return false;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        UploadSockLen size = sizeof(address);
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), size) != 0 ||
            ::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size) != 0 ||
            ::listen(listener_, 32) != 0) return false;
        port = ntohs(address.sin_port);
        thread_ = std::thread([this] { run(); });
        return true;
    }
    void stop() {
        stopped_.store(true);
        if (listener_ != kBadSocket) {
#ifdef _WIN32
            (void)::shutdown(listener_, SD_BOTH);
#else
            (void)::shutdown(listener_, SHUT_RDWR);
#endif
            close_upload_socket(listener_);
        }
        if (thread_.joinable()) thread_.join();
        listener_ = kBadSocket;
    }
    bool store(S3ObjectStore* config, ObjectStore* out) {
        if (!s3_object_store_init(config, "bucket", "us-east-1", "access", "secret", out)) return false;
        std::snprintf(config->endpoint, sizeof(config->endpoint), "http://127.0.0.1:%u/base", port);
        std::memcpy(config->session_token, token.c_str(), token.size() + 1u);
        config->path_style = true;
        return true;
    }
private:
    UploadSocket listener_ = kBadSocket;
    std::thread thread_;
    std::atomic<bool> stopped_{false};
    std::vector<uint8_t> expected_{std::vector<uint8_t>(kS3MaxPartBytes)};

    static std::string id(unsigned upload, bool escaped) {
        return "u" + std::to_string(upload) + (escaped ? "%2Bid%2F%3D%26x" : "+id/=&x");
    }
    static std::string result_xml(const char* root, const char* field,
                                   const std::string& value, const char* key = kUploadKey) {
        return std::string("<?xml version=\"1.0\" encoding=\"UTF-8\"?><") + root +
            " xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Bucket>bucket</Bucket><Key>" +
            key + "</Key><" + field + ">" + value + "</" + field + "></" + root + ">";
    }
    bool read_request(UploadSocket fd, std::string* head, std::string* small,
                       std::string* method, std::string* target, unsigned* upload, unsigned* part) {
        std::array<char, 65536> buffer{};
        while (head->find("\r\n\r\n") == std::string::npos && head->size() < 65536u) {
            const int n = static_cast<int>(::recv(fd, buffer.data(), static_cast<UploadIoSize>(buffer.size()), 0));
            if (n <= 0) return false;
            head->append(buffer.data(), static_cast<size_t>(n));
        }
        const size_t end = head->find("\r\n\r\n");
        if (end == std::string::npos) return false;
        const size_t first = head->find(' '), second = head->find(' ', first + 1u);
        if (first == std::string::npos || second == std::string::npos) return false;
        *method = head->substr(0, first);
        *target = head->substr(first + 1u, second - first - 1u);
        *upload = 0; *part = 0;
        for (unsigned i = 1; i <= creates && i <= uploads.size(); ++i)
            if (target->find("uploadId=" + id(i, true)) != std::string::npos) *upload = i;
        const size_t part_at = target->find("partNumber=");
        if (part_at != std::string::npos) *part = static_cast<unsigned>(std::strtoul(target->c_str() + part_at + 11u, nullptr, 10));
        const uint64_t length = std::strtoull(request_header(*head, "Content-Length").c_str(), nullptr, 10);
        if (length > kS3MaxPartBytes) return false;
        uint64_t read = 0;
        auto consume = [&](const char* bytes, size_t count) {
            if (count > length - read) return false;
            if (*part != 0) {
                const uint64_t offset = static_cast<uint64_t>(*part - 1u) * part_bytes + read;
                for (size_t i = 0; i < count; ++i)
                    if (static_cast<uint8_t>(bytes[i]) != pattern_byte(offset + i)) return false;
            } else if (small->size() + count <= 65536u) small->append(bytes, count);
            else return false;
            read += count;
            return true;
        };
        if (!consume(head->data() + end + 4u, head->size() - end - 4u)) return false;
        head->resize(end + 4u);
        while (read < length) {
            const int n = static_cast<int>(::recv(fd, buffer.data(),
                static_cast<UploadIoSize>(std::min<uint64_t>(buffer.size(), length - read)), 0));
            if (n <= 0 || !consume(buffer.data(), static_cast<size_t>(n))) return false;
        }
        if (*part != 0) {
            if (*upload == 0 || *part != uploads[*upload - 1u].parts + 1u) return false;
            const uint64_t offset = static_cast<uint64_t>(*part - 1u) * part_bytes;
            if (offset >= total || length != std::min<uint64_t>(part_bytes, total - offset)) return false;
            ++uploads[*upload - 1u].parts;
            uploads[*upload - 1u].bytes += length;
        }
        return validate_signature(*head, *method, *target, *small, *part, length);
    }
    bool validate_signature(const std::string& head, const std::string& method,
                              const std::string& target, const std::string& body,
                              unsigned part, uint64_t length) {
        const std::string authority = "127.0.0.1:" + std::to_string(port);
        if (request_header(head, "Host") != authority || request_header(head, "X-Amz-Security-Token") != token) return false;
        char hash[65];
        if (part != 0) {
            const uint64_t offset = static_cast<uint64_t>(part - 1u) * part_bytes;
            for (uint64_t i = 0; i < length; ++i) expected_[static_cast<size_t>(i)] = pattern_byte(offset + i);
            bolt::crypto::sha256_hex(expected_.data(), length, hash);
        } else bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(body.data()), body.size(), hash);
        if (request_header(head, "X-Amz-Content-Sha256") != hash) return false;
        const size_t query_at = target.find('?');
        const std::string path = target.substr(0, query_at);
        const std::string query = query_at == std::string::npos ? "" : target.substr(query_at + 1u);
        const std::string date = request_header(head, "X-Amz-Date");
        bolt::crypto::SigV4Request request{method.c_str(), authority.c_str(), path.c_str(), query.c_str(),
            "s3", "us-east-1", date.c_str(), hash, "access", "secret", token.c_str()};
        char auth[bolt::crypto::kSigV4MaxAuth], echoed[24];
        bolt::crypto::SigV4Result signed_result{auth, sizeof(auth), echoed, sizeof(echoed)};
        return path == kUploadPath && bolt::crypto::sigv4_sign(&request, &signed_result) &&
            request_header(head, "Authorization") == auth;
    }
    bool respond(UploadSocket fd, int status, const std::string& body, const std::string& extra = "") {
        return socket_send(fd, "HTTP/1.1 " + std::to_string(status) + " Result\r\nConnection: close\r\n" +
            extra + "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
    }
    bool complete(UploadSocket fd, const std::string& head, const std::string& body, unsigned upload) {
        ++completes;
        if (request_header(head, "If-None-Match") == "*") ++conditional_completes;
        std::string manifest = "<CompleteMultipartUpload>";
        for (unsigned part = 1; part <= uploads[upload - 1u].parts; ++part)
            manifest += "<Part><PartNumber>" + std::to_string(part) + "</PartNumber><ETag>&quot;p&amp;&lt;&gt;" +
                std::to_string(part) + "&quot;</ETag></Part>";
        manifest += "</CompleteMultipartUpload>";
        if (body != manifest || uploads[upload - 1u].bytes != total) return false;
        if (mode == UploadMode::Dropped) return true;
        if (mode == UploadMode::Error200) return respond(fd, 200, "<Error><Code>InternalError</Code></Error>");
        if (mode == UploadMode::Conditional412) return respond(fd, 412, "");
        if (mode == UploadMode::Conflict409) return respond(fd, 409, "");
        std::string result = result_xml("CompleteMultipartUploadResult", "ETag", "&quot;complete&quot;",
            mode == UploadMode::WrongKey ? "wrong" : kUploadKey);
        if (mode == UploadMode::MalformedComplete) result += "garbage";
        if (mode == UploadMode::Truncated)
            return socket_send(fd, "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(result.size()) +
                "\r\nConnection: close\r\n\r\n" + result.substr(0, result.size() / 2u));
        if (mode == UploadMode::ChunkedComplete) {
            const size_t middle = result.size() / 2u;
            char a[32], b[32];
            std::snprintf(a, sizeof(a), "%zx\r\n", middle);
            std::snprintf(b, sizeof(b), "%zx\r\n", result.size() - middle);
            return socket_send(fd, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n") &&
                socket_send(fd, std::string(a) + result.substr(0, middle) + "\r\n") &&
                socket_send(fd, std::string(b) + result.substr(middle) + "\r\n0\r\n\r\n");
        }
        return respond(fd, 200, result);
    }
    bool serve(UploadSocket fd) {
        std::string head, body, method, target;
        unsigned upload = 0, part = 0;
        if (!read_request(fd, &head, &body, &method, &target, &upload, &part)) return false;
        if (method == "POST" && target == std::string(kUploadPath) + "?uploads=") {
            if (++creates > uploads.size()) return false;
            std::string result = mode == UploadMode::MalformedInit ? "<InitiateMultipartUploadResult>" :
                result_xml("InitiateMultipartUploadResult", "UploadId", "u" + std::to_string(creates) + "+id/=&amp;x");
            if (mode == UploadMode::NumericId)
                result = result_xml("InitiateMultipartUploadResult", "UploadId", "u" + std::to_string(creates) + "&#43;id&#x2F;=&amp;x");
            if (mode == UploadMode::InitOversize)
                result = result_xml("InitiateMultipartUploadResult", "UploadId", std::string(kS3UploadIdCap, 'u'));
            if (mode == UploadMode::InitDuplicate)
                result.insert(result.find("</InitiateMultipartUploadResult>"), "<UploadId>other</UploadId>");
            if (mode == UploadMode::InitBadEntity)
                result = result_xml("InitiateMultipartUploadResult", "UploadId", "unknown&entity;");
            if (mode == UploadMode::InitBadDeclaration) result.replace(6, 7, "invalid");
            return respond(fd, 200, result);
        }
        if (method == "PUT" && target == kUploadPath) {
            ++puts;
            if (request_header(head, "If-None-Match") == "*") ++conditional_puts;
            if (mode == UploadMode::Conditional412) return respond(fd, 412, "");
            if (mode == UploadMode::Conflict409) return respond(fd, 409, "");
            if (mode == UploadMode::SmallPutServerError) return respond(fd, 503, "");
            accepted_put_body = body;
            if (mode == UploadMode::SmallPutDropped) return true;
            if (mode == UploadMode::SmallPutTruncated)
                return socket_send(fd, "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nx");
            if (mode == UploadMode::SmallPutEtag) return respond(fd, 200, "", "ETag: \"created-tag\"\r\n");
            return respond(fd, 200, "");
        }
        if (upload == 0) return false;
        const std::string query = part == 0 ? "?uploadId=" : "?partNumber=" + std::to_string(part) + "&uploadId=";
        if (target != std::string(kUploadPath) + query + id(upload, true)) return false;
        if (method == "DELETE") { ++aborts; return respond(fd, mode == UploadMode::AbortError ? 500 : 204, ""); }
        if (method == "POST" && part == 0) return complete(fd, head, body, upload);
        if (method != "PUT" || part == 0) return false;
        if ((mode == UploadMode::PartError || mode == UploadMode::AbortError) && part == 2)
            return respond(fd, 500, "");
        if (mode == UploadMode::MissingEtag) return respond(fd, 200, "");
        const std::string tag = mode == UploadMode::OversizeEtag ? std::string(kS3PartEtagCap, 'e') :
            "\"p&<>" + std::to_string(part) + "\"";
        return respond(fd, 200, "", "ETag: " + tag + "\r\n");
    }
    void run() {
        for (unsigned request = 0; request < 128u && !stopped_.load(); ++request) {
            const UploadSocket fd = ::accept(listener_, nullptr, nullptr);
            if (fd == kBadSocket) return;
#ifdef _WIN32
            DWORD timeout = 3000;
#else
            timeval timeout{3, 0};
#ifdef SO_NOSIGPIPE
            const int no_sigpipe = 1;
            (void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
#endif
            const bool timed = ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                reinterpret_cast<const char*>(&timeout), sizeof(timeout)) == 0 &&
                ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                reinterpret_cast<const char*>(&timeout), sizeof(timeout)) == 0;
            if (!timed || !serve(fd)) valid = false;
            close_upload_socket(fd);
        }
    }
};

struct PatternSource {
    std::vector<uint8_t> buffer;
    unsigned calls = 0;
    unsigned fail_on = 0, cancel_on = 0;
    bool cancel = false;
    explicit PatternSource(uint32_t size = kS3DefaultPartBytes) : buffer(size) {}
    static int read(void* context, uint64_t offset, uint32_t length, const uint8_t** data) noexcept {
        auto& source = *static_cast<PatternSource*>(context);
        ++source.calls;
        if (source.calls == source.fail_on) return kOsIoError;
        if (source.calls == source.cancel_on) source.cancel = true;
        if (length > source.buffer.size()) return kOsBufferSmall;
        for (uint32_t i = 0; i < length; ++i) source.buffer[i] = pattern_byte(offset + i);
        *data = source.buffer.data();
        return kOsOk;
    }
    static bool cancelled(void* context) noexcept { return static_cast<PatternSource*>(context)->cancel; }
};

int upload_pattern(UploadServer& server, PatternSource& pattern,
                     S3UploadResult* result, bool conditional = false) {
    S3ObjectStore config;
    ObjectStore store;
    if (!server.store(&config, &store)) return kOsBadArg;
    bolt::Arena workspace;
    S3UploadOptions options;
    options.part_bytes = server.part_bytes;
    options.if_absent = conditional;
    options.cancelled = PatternSource::cancelled;
    options.cancel_context = &pattern;
    S3UploadSource source{PatternSource::read, &pattern};
    return s3_upload_stream(&config, kUploadKey, server.total, &source, &options, &workspace, result);
}

TEST(S3MultipartUpload, ExistingContiguousPutCrossesSixteenMiBWithoutWholeBodyStaging) {
    UploadServer server;
    ASSERT_TRUE(server.start());
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    std::vector<uint8_t> body(static_cast<size_t>(kUploadBytes));
    for (uint64_t i = 0; i < kUploadBytes; ++i) body[static_cast<size_t>(i)] = pattern_byte(i);
    const int status = os_put(&store, kUploadKey, body.data(), body.size());
    server.stop();
    EXPECT_EQ(status, kOsOk);
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(server.creates, 1u);
    EXPECT_EQ(server.completes, 1u);
    EXPECT_EQ(server.aborts, 0u);
    EXPECT_EQ(server.uploads[0].parts, 3u);
    EXPECT_EQ(server.uploads[0].bytes, kUploadBytes);
}

TEST(S3MultipartUpload, StreamReusesOnePartBufferWithOpaqueIdsEtagsAndStsSigning) {
    UploadServer server;
    server.mode = UploadMode::ChunkedComplete;
    ASSERT_TRUE(server.start());
    PatternSource pattern;
    S3UploadResult result;
    const int status = upload_pattern(server, pattern, &result, true);
    server.stop();
    EXPECT_EQ(status, kOsOk);
    EXPECT_EQ(result.status, status);
    EXPECT_EQ(result.phase, S3UploadPhase::Done);
    EXPECT_EQ(result.bytes_uploaded, kUploadBytes);
    EXPECT_EQ(result.parts_uploaded, 3u);
    EXPECT_EQ(pattern.calls, 3u);
    EXPECT_EQ(pattern.buffer.size(), kS3DefaultPartBytes);
    EXPECT_STREQ(result.upload_id, "u1+id/=&x");
    EXPECT_EQ(server.conditional_completes, 1u);
    EXPECT_TRUE(server.valid);
}

TEST(S3MultipartUpload, PrecompletionFailuresAbortOnceAndRetainBothOutcomes) {
    for (UploadMode mode : {UploadMode::PartError, UploadMode::AbortError,
                            UploadMode::MissingEtag, UploadMode::OversizeEtag}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        PatternSource pattern;
        S3UploadResult result;
        EXPECT_EQ(upload_pattern(server, pattern, &result), kOsIoError);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(server.aborts, 1u);
        EXPECT_EQ(server.completes, 0u);
        EXPECT_EQ(result.abort_status, mode == UploadMode::AbortError ? kOsIoError : kOsOk);
        EXPECT_EQ(result.remote_state_uncertain, mode == UploadMode::AbortError);
        EXPECT_STREQ(result.upload_id, "u1+id/=&x");
        EXPECT_FALSE(result.completion_attempted);
    }
}

TEST(S3MultipartUpload, SourceFailureAndCancellationSuppressPartsAndCompletion) {
    for (bool cancel : {false, true}) {
        UploadServer server;
        ASSERT_TRUE(server.start());
        PatternSource pattern;
        if (cancel) pattern.cancel_on = 2;
        else pattern.fail_on = 2;
        S3UploadResult result;
        EXPECT_EQ(upload_pattern(server, pattern, &result), cancel ? kOsCancelled : kOsIoError);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(result.phase, S3UploadPhase::Source);
        EXPECT_EQ(result.parts_uploaded, 1u);
        EXPECT_EQ(server.uploads[0].parts, 1u);
        EXPECT_EQ(server.aborts, 1u);
        EXPECT_EQ(server.completes, 0u);
    }
}

TEST(S3MultipartUpload, UnverifiedCompletionIsUncertainWithoutAbortOrRetry) {
    for (UploadMode mode : {UploadMode::Error200, UploadMode::Truncated, UploadMode::Dropped,
                            UploadMode::Conflict409, UploadMode::WrongKey, UploadMode::MalformedComplete}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        PatternSource pattern;
        S3UploadResult result;
        EXPECT_EQ(upload_pattern(server, pattern, &result, true), kOsWriteUncertain);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_TRUE(result.remote_state_uncertain);
        EXPECT_TRUE(result.completion_attempted);
        EXPECT_EQ(result.phase, S3UploadPhase::Complete);
        EXPECT_EQ(server.creates, 1u);
        EXPECT_EQ(server.completes, 1u);
        EXPECT_EQ(server.conditional_completes, 1u);
        EXPECT_EQ(server.aborts, 0u);
        EXPECT_STREQ(result.upload_id, "u1+id/=&x");
    }
}

TEST(S3MultipartUpload, ConditionalCompletionPublishesPreconditionAndMapsOnly412ToExists) {
    UploadServer server;
    server.mode = UploadMode::Conditional412;
    ASSERT_TRUE(server.start());
    PatternSource pattern;
    S3UploadResult result;
    EXPECT_EQ(upload_pattern(server, pattern, &result, true), kOsExists);
    server.stop();
    EXPECT_EQ(result.http_status, 412u);
    EXPECT_FALSE(result.remote_state_uncertain);
    EXPECT_EQ(server.conditional_completes, 1u);
    EXPECT_EQ(server.completes, 1u);
    EXPECT_EQ(server.aborts, 0u);
    EXPECT_TRUE(server.valid);
}

TEST(S3MultipartUpload, InvalidGeometryBudgetAndEarlyCancellationHaveNoRemoteEffect) {
    UploadServer server;
    ASSERT_TRUE(server.start());
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    PatternSource pattern;
    S3UploadSource source{PatternSource::read, &pattern};
    S3UploadResult result;
    bolt::Arena workspace;
    for (uint32_t size : {0u, kS3MinPartBytes - 1u, kS3MaxPartBytes + 1u}) {
        S3UploadOptions options;
        options.part_bytes = size;
        EXPECT_EQ(s3_upload_stream(&config, kUploadKey, kUploadBytes, &source, &options, &workspace, &result), kOsBadArg);
    }
    S3UploadOptions options;
    EXPECT_EQ(s3_upload_stream(&config, kUploadKey,
        static_cast<uint64_t>(kS3MaxUploadParts + 1u) * options.part_bytes,
        &source, &options, &workspace, &result), kOsBufferSmall);
    bolt::Arena tiny{bolt::ArenaConfig{128u, 128u}};
    tiny.set_growth_enabled(false);
    EXPECT_EQ(s3_upload_stream(&config, kUploadKey, kUploadBytes, &source, &options, &tiny, &result), kOsBufferSmall);
    config.create_only = kS3CreateOnlyGoogGeneration;
    options.if_absent = true;
    EXPECT_EQ(s3_upload_stream(&config, kUploadKey, kUploadBytes, &source, &options, &workspace, &result), kOsNotImplemented);
    config.create_only = kS3CreateOnlyIfNoneMatch;
    pattern.cancel = true;
    options.cancelled = PatternSource::cancelled;
    options.cancel_context = &pattern;
    EXPECT_EQ(s3_upload_stream(&config, kUploadKey, kUploadBytes, &source, &options, &workspace, &result), kOsCancelled);
    server.stop();
    EXPECT_EQ(pattern.calls, 0u);
    EXPECT_EQ(server.creates, 0u);
    EXPECT_EQ(server.puts, 0u);
}

TEST(S3MultipartUpload, MalformedInitiationNeverUsesAnUnverifiedUploadId) {
    for (UploadMode mode : {UploadMode::MalformedInit, UploadMode::InitOversize,
            UploadMode::InitDuplicate, UploadMode::InitBadEntity, UploadMode::InitBadDeclaration}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        PatternSource pattern;
        S3UploadResult result;
        EXPECT_EQ(upload_pattern(server, pattern, &result), kOsIoError);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(pattern.calls, 0u);
        EXPECT_TRUE(result.remote_state_uncertain);
        EXPECT_EQ(result.phase, S3UploadPhase::Create);
        EXPECT_STREQ(result.upload_id, "");
        EXPECT_EQ(server.creates, 1u);
        EXPECT_EQ(server.aborts, 0u);
        EXPECT_EQ(server.completes, 0u);
    }
}

TEST(S3MultipartUpload, NumericUploadIdEntitiesAndPartBoundaryGeometry) {
    for (uint32_t size : {kS3MinPartBytes, kS3MaxPartBytes}) {
        UploadServer server;
        server.mode = UploadMode::NumericId;
        server.part_bytes = size;
        server.total = static_cast<uint64_t>(size) + 1u;
        ASSERT_TRUE(server.start());
        PatternSource pattern(size);
        S3UploadResult result;
        EXPECT_EQ(upload_pattern(server, pattern, &result), kOsOk);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(result.parts_uploaded, 2u);
        EXPECT_EQ(result.bytes_uploaded, server.total);
        EXPECT_STREQ(result.upload_id, "u1+id/=&x");
    }
}

TEST(S3MultipartUpload, ExistingConditionalPutUsesAtomicCompletion) {
    UploadServer server;
    server.mode = UploadMode::Conditional412;
    ASSERT_TRUE(server.start());
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    std::vector<uint8_t> body(static_cast<size_t>(kUploadBytes));
    for (uint64_t i = 0; i < kUploadBytes; ++i) body[static_cast<size_t>(i)] = pattern_byte(i);
    const int status = os_put_if_absent(&store, kUploadKey, body.data(), body.size());
    server.stop();
    EXPECT_EQ(status, kOsExists);
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(server.conditional_completes, 1u);
    EXPECT_EQ(server.completes, 1u);
    EXPECT_EQ(server.aborts, 0u);
}

TEST(S3MultipartUpload, ZeroByteStreamUsesSingleConditionalPutWithoutSourceCallback) {
    UploadServer server;
    server.total = 0;
    ASSERT_TRUE(server.start());
    PatternSource pattern;
    S3UploadResult result;
    EXPECT_EQ(upload_pattern(server, pattern, &result, true), kOsOk);
    server.stop();
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(pattern.calls, 0u);
    EXPECT_EQ(server.puts, 1u);
    EXPECT_EQ(server.creates, 0u);
}

TEST(S3MultipartUpload, ExistingSmallAndEmptyPutsRetainSingleRequestShape) {
    UploadServer server;
    ASSERT_TRUE(server.start());
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    const uint8_t bytes[] = {'s', 'm', 'a', 'l', 'l'};
    EXPECT_EQ(os_put(&store, kUploadKey, bytes, sizeof(bytes)), kOsOk);
    EXPECT_EQ(os_put(&store, kUploadKey, nullptr, 0), kOsOk);
    server.stop();
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(server.puts, 2u);
    EXPECT_EQ(server.creates, 0u);
}

TEST(S3MultipartUpload, AcceptedSmallConditionalPutWithLostReplyIsUncertainWithoutRetry) {
    for (UploadMode mode : {UploadMode::SmallPutDropped, UploadMode::SmallPutTruncated}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        S3ObjectStore config;
        ObjectStore store;
        ASSERT_TRUE(server.store(&config, &store));
        constexpr char metadata[] = "{\"format-version\":2,\"current-snapshot-id\":17}";
        EXPECT_EQ(os_put_if_absent(&store, kUploadKey,
            reinterpret_cast<const uint8_t*>(metadata), sizeof(metadata) - 1u), kOsWriteUncertain);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(server.accepted_put_body, metadata);
        EXPECT_EQ(server.puts, 1u);
        EXPECT_EQ(server.conditional_puts, 1u);
        EXPECT_EQ(server.creates, 0u);
        EXPECT_EQ(server.aborts, 0u);
    }
}

TEST(S3MultipartUpload, SmallConditionalPutDistinguishesDefinitive412FromAmbiguous409And503) {
    for (UploadMode mode : {UploadMode::Conditional412, UploadMode::Conflict409,
                             UploadMode::SmallPutServerError}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        S3ObjectStore config;
        ObjectStore store;
        ASSERT_TRUE(server.store(&config, &store));
        const int expected = mode == UploadMode::Conditional412 ? kOsExists : kOsWriteUncertain;
        EXPECT_EQ(os_put_if_absent(&store, kUploadKey, nullptr, 0), expected);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(server.puts, 1u);
        EXPECT_EQ(server.conditional_puts, 1u);
        EXPECT_EQ(server.creates, 0u);
    }
}

TEST(S3MultipartUpload, ExplicitCleanupUsesTheRetainedOpaqueIdWithoutRetry) {
    UploadServer server;
    server.mode = UploadMode::Conditional412;
    ASSERT_TRUE(server.start());
    PatternSource pattern;
    S3UploadResult result;
    EXPECT_EQ(upload_pattern(server, pattern, &result, true), kOsExists);
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    EXPECT_EQ(s3_abort_multipart(&config, kUploadKey, result.upload_id), kOsOk);
    server.stop();
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(server.creates, 1u);
    EXPECT_EQ(server.completes, 1u);
    EXPECT_EQ(server.aborts, 1u);
}

TEST(S3MultipartUpload, ConcurrentCallsKeepIdsProgressAndSourceBuffersIndependent) {
    UploadServer server;
    ASSERT_TRUE(server.start());
    S3ObjectStore config;
    ObjectStore store;
    ASSERT_TRUE(server.store(&config, &store));
    std::array<S3UploadResult, 2> results;
    std::array<int, 2> statuses{};
    std::array<std::thread, 2> clients;
    for (size_t i = 0; i < clients.size(); ++i) clients[i] = std::thread([&, i] {
        PatternSource pattern;
        bolt::Arena workspace;
        S3UploadSource source{PatternSource::read, &pattern};
        S3UploadOptions options;
        statuses[i] = s3_upload_stream(&config, kUploadKey, kUploadBytes, &source, &options, &workspace, &results[i]);
    });
    for (auto& client : clients) client.join();
    server.stop();
    EXPECT_TRUE(server.valid);
    EXPECT_EQ(server.creates, 2u);
    EXPECT_EQ(server.completes, 2u);
    EXPECT_STRNE(results[0].upload_id, results[1].upload_id);
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(statuses[i], kOsOk);
        EXPECT_EQ(results[i].bytes_uploaded, kUploadBytes);
        EXPECT_EQ(results[i].parts_uploaded, 3u);
    }
}
}  // namespace

// G2ICE-349: a conditional create reports the version its response acknowledged; without an ETag the version is zeroed.
TEST(S3MultipartUpload, VersionedConditionalPutReportsTheAcknowledgedEtag) {
    for (UploadMode mode : {UploadMode::SmallPutEtag, UploadMode::Success, UploadMode::Conditional412}) {
        SCOPED_TRACE(static_cast<unsigned>(mode));
        UploadServer server;
        server.mode = mode;
        ASSERT_TRUE(server.start());
        S3ObjectStore config;
        ObjectStore store;
        ASSERT_TRUE(server.store(&config, &store));
        const uint8_t bytes[] = {'s', 'm', 'a', 'l', 'l'};
        ObjectVersion v{};
        std::strcpy(v.etag, "\"stale\"");
        const int rc = os_put_if_absent_versioned(&store, kUploadKey, bytes, sizeof(bytes), &v);
        server.stop();
        EXPECT_TRUE(server.valid);
        EXPECT_EQ(server.conditional_puts, 1u);
        if (mode == UploadMode::Conditional412) { EXPECT_EQ(rc, kOsExists); EXPECT_EQ(v.etag[0], 0); continue; }
        EXPECT_EQ(rc, kOsOk);
        if (mode == UploadMode::SmallPutEtag) { EXPECT_STREQ(v.etag, "\"created-tag\""); EXPECT_EQ(v.size, sizeof(bytes)); }
        else EXPECT_EQ(v.etag[0], 0) << "no ETag in the response: zeroed, the caller HEADs";
    }
}
