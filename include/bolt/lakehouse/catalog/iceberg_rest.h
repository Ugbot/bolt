// bolt/lakehouse/catalog/iceberg_rest.h — Apache Iceberg REST Catalog.
//
// Speaks the full Apache REST Catalog spec at /v1[/{prefix}]:
//   - /config discovery (defaults + overrides → prefix / default catalog)
//   - namespace CRUD + table CRUD (list/load/create/commit/drop/register/rename)
//   - multi-table transactions (/transactions/commit)
//   - views (create/drop/load/commit/rename)
//   - snapshots (list / delete_snapshots / cherry_pick / rollback)
//   - branches + tags (MANAGE_SNAPSHOTS set/remove-snapshot-ref updates)
//   - statistics + partition statistics updates
//   - pageToken + pageSize pagination on every list endpoint
//   - vended credentials (X-Iceberg-Access-Delegation)
//   - CDC feed (/cdc) when advertised
// Auth modes: none, bearer, basic, oauth2 client-credentials, sigv4 (API GW).
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, bounded.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/catalog/secret.h"
#include "bolt/net/bolt_http_client.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg_rest {

// Auth mode discriminator for Config::auth_mode.
enum : uint8_t {
    kAuthNone   = 0u,
    kAuthBearer = 1u,
    kAuthOAuth2 = 2u,
    kAuthSigV4  = 3u,
    kAuthBasic  = 4u,
};

static constexpr uint32_t kIrcMaxRef       = 128u;   // branch/tag/view name
static constexpr uint32_t kIrcMaxScope     = 256u;   // OAuth2 scope string
static constexpr uint32_t kIrcMaxCatalog   = 128u;   // discovered default catalog
static constexpr uint32_t kIrcMaxSnapshots = 256u;   // bounded snapshot listing

struct Config {
    char    base_url[512];           // "https://host[:port]" — NO trailing slash
    char    prefix[128];             // e.g. "" or "ws/main"
    uint8_t auth_mode;               // see kAuth* above
    Secret  bearer_token;            // kAuthBearer
    char    oauth_token_url[512];    // kAuthOAuth2 token endpoint
    Secret  oauth_client_id;
    Secret  oauth_client_secret;
    char    oauth_scope[kIrcMaxScope];  // optional OAuth2 scope (may be empty)
    Secret  basic_user;              // kAuthBasic
    Secret  basic_password;
    char    aws_region[32];          // kAuthSigV4
    char    aws_service[32];         // "execute-api" default (no crypto include)
    Secret  aws_access_key;
    Secret  aws_secret_key;
    Secret  aws_session_token;       // optional STS token (may be empty)
    uint8_t vended_credentials;      // 1 → send X-Iceberg-Access-Delegation
};

// Bind `out` to a freshly-opened catalog. Allocates impl from `arena`.
bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// Run GET /v1/config discovery and apply overrides/defaults to `impl`.
// Returns kCatOk / kCatNotImplemented (server lacks /config) / kCatIoError.
int discover_config(Catalog* cat) noexcept;

// --- Snapshots ---
struct SnapshotRef {
    int64_t snapshot_id;
    int64_t parent_snapshot_id;     // -1 if absent
    int64_t sequence_number;        // -1 if absent
    char    operation[32];          // "append" / "overwrite" / ...
};

// --- Test helpers (white-box). Stable for unit tests. ---

struct Impl {
    Config        cfg;
    bolt::Arena*  arena;
    char          cached_token[512];
    int64_t       token_expiry_epoch;
    // Populated by discover_config():
    char          default_catalog[kIrcMaxCatalog];
    uint8_t        cdc_supported;       // 1 if /config advertised the CDC feed
    uint8_t        config_discovered;   // 1 once /config applied
};

// Build a `GET /v1/{prefix}/namespaces` request (no send). Threads pageToken /
// pageSize into the query when `page_token`/`page_size` are set.
bool irc_build_list_namespaces_request(const Impl* impl,
                                       bolt::net::HttpRequest* out) noexcept;
bool irc_build_list_namespaces_request_paged(const Impl* impl,
                                             const char* page_token,
                                             uint32_t page_size,
                                             bolt::net::HttpRequest* out) noexcept;

// Build the OAuth2 client-credentials token-exchange request (no send).
bool irc_build_oauth2_token_request(const Impl* impl,
                                    bolt::net::HttpRequest* out,
                                    char* body_buf, uint32_t body_cap) noexcept;

// Build a multi-table transaction commit body into `out` (returns length, or 0
// on overflow / bad arg). `table_jsons[i]` is a pre-built per-table commit JSON
// object string ({"identifier":...,"requirements":[...],"updates":[...]}).
uint32_t irc_build_tx_commit_body(const char* const* table_jsons,
                                  uint32_t n, char* out,
                                  uint32_t cap) noexcept;

// Build a single-table set-snapshot-ref update object (branch or tag) into
// `out`. `type` is "branch" or "tag". Returns length or 0 on overflow.
uint32_t irc_build_set_snapshot_ref_update(const char* ref_name,
                                           const char* type,
                                           int64_t snapshot_id, char* out,
                                           uint32_t cap) noexcept;

// Standard RFC4648 base64 of `src` (NUL-terminated out). Returns length or 0
// on overflow. Exposed for the Basic-auth wire test.
uint32_t irc_base64_std(const uint8_t* src, uint32_t n, char* out,
                        uint32_t cap) noexcept;

// Parse a `{"namespaces": [["ns1"], ...]}` body into CatalogName[].
// If `out_next_token` is non-null, copies any `next-page-token` into it.
bool irc_parse_namespaces(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                          CatalogName* out, uint32_t cap,
                          uint32_t* out_n) noexcept;
bool irc_parse_namespaces_paged(const uint8_t* body, uint32_t len,
                                bolt::Arena* arena, CatalogName* out,
                                uint32_t cap, uint32_t* out_n,
                                char* out_next_token,
                                uint32_t token_cap) noexcept;

// Parse a list-snapshots body into SnapshotRef[].
bool irc_parse_snapshots(const uint8_t* body, uint32_t len, bolt::Arena* arena,
                         SnapshotRef* out, uint32_t cap,
                         uint32_t* out_n) noexcept;

// --- High-level operations beyond the base CatalogVT ---

// Commit a multi-table transaction. `table_jsons[i]` per F.1; see
// irc_build_tx_commit_body. Returns kCat* code.
int commit_transaction(Catalog* cat, const char* const* table_jsons,
                       uint32_t n) noexcept;

// List snapshots for a table.
int list_snapshots(Catalog* cat, const char* ns, const char* table,
                   SnapshotRef* out, uint32_t cap, uint32_t* out_n) noexcept;

// Register an externally-written table (adopt metadata.json at `metadata_loc`).
int register_table(Catalog* cat, const char* ns, const char* table,
                   const char* metadata_loc) noexcept;

// Rename a table (POST /v1/{prefix}/tables/rename).
int rename_table(Catalog* cat, const char* from_ns, const char* from_name,
                 const char* to_ns, const char* to_name) noexcept;

// Set / remove a branch or tag via a MANAGE_SNAPSHOTS update commit.
int set_snapshot_ref(Catalog* cat, const char* ns, const char* table,
                     const char* ref_name, const char* type,
                     int64_t snapshot_id) noexcept;
int remove_snapshot_ref(Catalog* cat, const char* ns, const char* table,
                        const char* ref_name) noexcept;

// Roll a table back to a known snapshot (set-snapshot-ref on "main").
int rollback_to_snapshot(Catalog* cat, const char* ns, const char* table,
                         int64_t snapshot_id) noexcept;

// Cherry-pick a snapshot onto the current state (cherrypick-snapshot update).
int cherry_pick(Catalog* cat, const char* ns, const char* table,
                int64_t snapshot_id) noexcept;

// Delete a set of snapshots by id (remove-snapshots update).
int delete_snapshots(Catalog* cat, const char* ns, const char* table,
                     const int64_t* snapshot_ids, uint32_t n) noexcept;

// set-statistics / remove-statistics / set-partition-statistics updates.
// `statistics_file_json` is a pre-built statistics-file object string.
int set_statistics(Catalog* cat, const char* ns, const char* table,
                   int64_t snapshot_id,
                   const char* statistics_file_json) noexcept;
int remove_statistics(Catalog* cat, const char* ns, const char* table,
                      int64_t snapshot_id) noexcept;
int set_partition_statistics(Catalog* cat, const char* ns, const char* table,
                             const char* partition_statistics_file_json) noexcept;

// --- Views ---
int create_view(Catalog* cat, const char* ns, const char* name,
                const char* schema_json, const char* sql) noexcept;
int drop_view(Catalog* cat, const char* ns, const char* name) noexcept;
int load_view(Catalog* cat, const char* ns, const char* name, char* out_loc,
              uint32_t cap) noexcept;
int rename_view(Catalog* cat, const char* from_ns, const char* from_name,
                const char* to_ns, const char* to_name) noexcept;

// --- CDC feed (only when discover_config() set cdc_supported) ---
int read_cdc(Catalog* cat, const char* ns, const char* table,
             const uint8_t** out_body, uint32_t* out_len) noexcept;

}  // namespace iceberg_rest
}  // namespace lakehouse
}  // namespace bolt
