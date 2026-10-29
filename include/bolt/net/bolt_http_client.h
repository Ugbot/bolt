// bolt/net/bolt_http_client.h — minimal HTTP/1.1 client used by lakehouse
// catalog backends (Iceberg REST, Unity, Glue, …). Plain TCP + bolt::net::TlsSocket
// for HTTPS. Bounded everything; arena-owned response body. PODs only.
#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace net {

static constexpr uint32_t kHttpMaxUrl     = 2048u;
static constexpr uint32_t kHttpMaxHeader  = 2048u;
static constexpr uint32_t kHttpMaxHeaders = 32u;
static constexpr uint32_t kHttpMaxBody    = 16u * 1024u * 1024u;

struct HttpHeader {
    char name[128];
    char value[kHttpMaxHeader];
};

struct HttpRequest {
    char           method[8];                    // "GET" / "POST" / ...
    char           url[kHttpMaxUrl];              // "https://host:port/path?q=1"
    HttpHeader     headers[kHttpMaxHeaders];
    uint32_t       header_count;
    const uint8_t* body;
    uint32_t       body_len;
    uint16_t       timeout_ms;                    // 0 → default
    uint32_t       max_response_body;             // 0 → kHttpMaxBody; otherwise <= it
};

struct HttpResponse {
    int            status;                        // HTTP status code
    HttpHeader     headers[kHttpMaxHeaders];
    uint32_t       header_count;
    const uint8_t* body;                          // arena-owned
    uint32_t       body_len;
};

// Parse `url` into components. Returns true on success.
// out_https=true for "https://", false for "http://".
bool http_url_parse(const char* url, bool* out_https,
                    char* out_host, uint32_t host_cap,
                    uint16_t* out_port,
                    char* out_path, uint32_t path_cap) noexcept;

// Host authority used on the wire and by request signers. Preserves nondefault
// ports and IPv6 brackets. The transport currently resolves IPv4 endpoints.
bool http_url_authority(const char* url, char* out, uint32_t cap) noexcept;

// Send `req` and read full response. body buffer arena-owned. 0 on success,
// negative on error; a failed response never publishes a partial body. No retry.
// timeout_ms is a total connect/send/receive deadline after synchronous DNS;
// accepted values are 1..60000 ms, or zero for 10000 ms. Response framing and
// declared/decoded body budgets are validated, without truncating fields.
// Response heads use kHttpResponseHeadCap (8192 bytes); wire staging is bounded
// by the body budget plus 64 KiB of framing overhead. Unsupported transfer
// codings fail explicitly. Requests must be zero-initialized before filling.
// Host/Connection are transport-owned; request Transfer-Encoding is unsupported.
// An explicit Content-Length must equal body_len; it is emitted exactly once.
// HttpHeader/HttpRequest layouts changed: consumers must rebuild together.
int http_send(bolt::Arena* arena, const HttpRequest* req,
              HttpResponse* out) noexcept;

// Streamed response (G2ICE-381): the same request rules as http_send, but the body is handed to on_body in pieces of at most
// buf_cap bytes from the caller's buffer instead of being materialized, so a large ranged GET is one request. on_head sees the
// parsed head (body null) before any body byte and may refuse it. The body must be Content-Length framed and at most max_body
// (chunked, close-delimited or larger bodies are refused after on_head, before any body byte); a head with no body (HEAD, 204,
// 304) calls on_head only.
// timeout_ms bounds connect, the head, and each wait for more body bytes (time without progress), not the whole transfer.
// 0: every declared byte delivered; -4: a callback refused; -6: unsupported framing; other negatives: transport or a short body.
struct HttpBodyStream {
    bool (*on_head)(void* ctx, const HttpResponse* head) noexcept;
    bool (*on_body)(void* ctx, const uint8_t* data, uint32_t n) noexcept;
    void*    ctx;
    uint8_t* buf;
    uint32_t buf_cap;
    uint32_t _pad;
    uint64_t max_body;
};
int http_send_stream(const HttpRequest* req, HttpResponse* out, const HttpBodyStream* stream) noexcept;

// Lookup header by case-insensitive name. Returns value or nullptr.
const char* http_header_get(const HttpHeader* hdrs, uint32_t n,
                            const char* name) noexcept;

// Append a header to req (bounded). Returns false on overflow.
bool http_request_add_header(HttpRequest* req, const char* name,
                             const char* value) noexcept;

}  // namespace net
}  // namespace bolt
