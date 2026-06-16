// bolt/net/bolt_http_client.h — minimal HTTP/1.1 client used by lakehouse
// catalog backends (Iceberg REST, Unity, Glue, …). Plain TCP + bolt::net::TlsSocket
// for HTTPS. Bounded everything; arena-owned response body. PODs only.
#pragma once

#include <cstdint>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace net {

static constexpr uint32_t kHttpMaxUrl     = 2048u;
static constexpr uint32_t kHttpMaxHeader  = 512u;
static constexpr uint32_t kHttpMaxHeaders = 32u;
static constexpr uint32_t kHttpMaxBody    = 16u * 1024u * 1024u;

struct HttpHeader {
    char name[128];
    char value[512];
};

struct HttpRequest {
    char           method[8];                    // "GET" / "POST" / ...
    char           url[kHttpMaxUrl];              // "https://host:port/path?q=1"
    HttpHeader     headers[kHttpMaxHeaders];
    uint32_t       header_count;
    const uint8_t* body;
    uint32_t       body_len;
    uint16_t       timeout_ms;                    // 0 → default
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

// Send `req` and read full response. body buffer arena-owned. 0 on success,
// negative on error.
int http_send(bolt::Arena* arena, const HttpRequest* req,
              HttpResponse* out) noexcept;

// Lookup header by case-insensitive name. Returns value or nullptr.
const char* http_header_get(const HttpHeader* hdrs, uint32_t n,
                            const char* name) noexcept;

// Append a header to req (bounded). Returns false on overflow.
bool http_request_add_header(HttpRequest* req, const char* name,
                             const char* value) noexcept;

}  // namespace net
}  // namespace bolt
