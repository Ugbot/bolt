// bolt/lakehouse/catalog/nessie.h — Project Nessie catalog (Iceberg REST + git-like refs).
//
// Nessie exposes an Iceberg REST Catalog surface (under an `/iceberg` prefix)
// plus a native v2 tree API for git-style reference workflows: commit, merge,
// transplant (cherry-pick a range of commits), and compare-references (diff).
// `open()` binds the shared iceberg_rest client for the Iceberg surface; the
// Nessie-native calls below are shaped here and sent over the bolt HTTP client.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, bounded.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

namespace bolt {
namespace lakehouse {
namespace nessie {

// Iceberg REST prefix Nessie serves the catalog surface under.
static constexpr const char* kNessieIcebergPrefix = "iceberg";

struct Config {
    char   base_url[512];      // "https://host[:port]" — NO trailing slash
    Secret bearer_token;       // optional (empty → no auth)
    char   reference[128];     // default branch, e.g. "main"
};

// Bind `out` to a Nessie catalog backed by the Iceberg REST client.
bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --- Nessie-native v2 tree API ---
//
// Reference: GET /api/v2/trees/{ref} → resolves a named ref to its hash.
// `out_hash` receives the ref's current commit hash (NUL-terminated).
int nessie_get_reference(Catalog* cat, const char* ref_name, char* out_hash,
                         uint32_t cap) noexcept;

// Commit: POST /api/v2/trees/{branch}@{expected_hash}/history/commit.
// `operations_json` is a pre-built `"operations":[...]` array string (may be
// "[]" for an empty commit). `expected_hash` may be empty (HEAD).
int nessie_commit(Catalog* cat, const char* branch, const char* expected_hash,
                  const char* message, const char* operations_json) noexcept;

// Merge: POST /api/v2/trees/{to}/history/merge — merges `from_ref`@`from_hash`
// into branch `to`. `from_hash` may be empty (HEAD of from_ref).
int nessie_merge(Catalog* cat, const char* from_ref, const char* from_hash,
                 const char* to_branch) noexcept;

// Transplant: POST /api/v2/trees/{to}/history/transplant — cherry-picks the
// listed commit hashes from `from_ref` onto branch `to`.
int nessie_transplant(Catalog* cat, const char* from_ref, const char* to_branch,
                      const char* const* hashes, uint32_t n) noexcept;

// Compare references: GET /api/v2/trees/{from}/diff/{to}. `out_body` points at
// the arena-owned diff JSON.
int nessie_compare_references(Catalog* cat, const char* from_ref,
                              const char* to_ref, const uint8_t** out_body,
                              uint32_t* out_len) noexcept;

// --- Request-shaping helpers (white-box; stable for unit tests) ---

bool nessie_build_get_reference(const char* base_url, const char* bearer_token,
                                const char* ref_name,
                                bolt::net::HttpRequest* out) noexcept;

bool nessie_build_commit(const char* base_url, const char* bearer_token,
                         const char* branch, const char* expected_hash,
                         const char* message, const char* operations_json,
                         bolt::net::HttpRequest* out, char* body_buf,
                         uint32_t body_cap) noexcept;

bool nessie_build_merge(const char* base_url, const char* bearer_token,
                        const char* from_ref, const char* from_hash,
                        const char* to_branch, bolt::net::HttpRequest* out,
                        char* body_buf, uint32_t body_cap) noexcept;

bool nessie_build_transplant(const char* base_url, const char* bearer_token,
                             const char* from_ref, const char* to_branch,
                             const char* const* hashes, uint32_t n,
                             bolt::net::HttpRequest* out, char* body_buf,
                             uint32_t body_cap) noexcept;

bool nessie_build_compare_references(const char* base_url,
                                     const char* bearer_token,
                                     const char* from_ref, const char* to_ref,
                                     bolt::net::HttpRequest* out) noexcept;

}  // namespace nessie
}  // namespace lakehouse
}  // namespace bolt
