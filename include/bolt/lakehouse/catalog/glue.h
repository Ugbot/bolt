// bolt/lakehouse/catalog/glue.h — AWS Glue Data Catalog.
// JSON-over-HTTPS at glue.<region>.amazonaws.com, SigV4-signed.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace glue {

struct Config {
    char   region[32];
    Secret access_key;
    Secret secret_key;
    Secret session_token;   // may be empty
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// Internal impl (visible for tests).
struct Impl {
    Config       cfg;
    bolt::Arena* arena;
};

// Sign a Glue HTTP request in place (adds Authorization, X-Amz-Date, …).
// Uses current epoch as the date.
bool glue_sign_request(Impl* impl, bolt::net::HttpRequest* req) noexcept;

// Same, but with a fixed amz_date for reproducibility.
bool glue_sign_request_with_date(Impl* impl, bolt::net::HttpRequest* req,
                                 const char* amz_date) noexcept;

}  // namespace glue
}  // namespace lakehouse
}  // namespace bolt
