// Tests for the Iceberg REST catalog wire shape + JSON parse.

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

using bolt::lakehouse::CatalogName;
using bolt::lakehouse::iceberg_rest::Config;
using bolt::lakehouse::iceberg_rest::Impl;

TEST(IcebergRest, BuildListNamespacesRequest_BearerAuth) {
    bolt::Arena arena;
    Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://catalog.example.com", sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.prefix, "ws/main", sizeof(cfg.prefix) - 1u);
    cfg.auth_mode = 1;
    std::strncpy(cfg.bearer_token.value, "tok-abc",
                 sizeof(cfg.bearer_token.value) - 1u);

    Impl impl;
    std::memset(&impl, 0, sizeof(impl));
    impl.cfg = cfg;
    impl.arena = &arena;

    bolt::net::HttpRequest req;
    ASSERT_TRUE(bolt::lakehouse::iceberg_rest::irc_build_list_namespaces_request(
        &impl, &req));
    EXPECT_STREQ(req.method, "GET");
    EXPECT_NE(std::strstr(req.url, "/v1/ws/main/namespaces"), nullptr);

    bool found_auth = false;
    for (uint32_t i = 0; i < req.header_count; ++i) {
        if (std::strcmp(req.headers[i].name, "Authorization") == 0) {
            EXPECT_STREQ(req.headers[i].value, "Bearer tok-abc");
            found_auth = true;
        }
    }
    EXPECT_TRUE(found_auth);
}

TEST(IcebergRest, ParseNamespaces) {
    bolt::Arena arena;
    const char* body =
        "{\"namespaces\":[[\"dev\"],[\"prod\"],[\"sandbox\"]]}";
    CatalogName out[8];
    uint32_t n = 0;
    ASSERT_TRUE(bolt::lakehouse::iceberg_rest::irc_parse_namespaces(
        reinterpret_cast<const uint8_t*>(body),
        static_cast<uint32_t>(std::strlen(body)),
        &arena, out, 8, &n));
    ASSERT_EQ(n, 3u);
    EXPECT_STREQ(out[0].name, "dev");
    EXPECT_STREQ(out[1].name, "prod");
    EXPECT_STREQ(out[2].name, "sandbox");
}

TEST(IcebergRest, ParseNamespaces_Empty) {
    bolt::Arena arena;
    const char* body = "{\"namespaces\":[]}";
    CatalogName out[4];
    uint32_t n = 99;
    ASSERT_TRUE(bolt::lakehouse::iceberg_rest::irc_parse_namespaces(
        reinterpret_cast<const uint8_t*>(body),
        static_cast<uint32_t>(std::strlen(body)),
        &arena, out, 4, &n));
    EXPECT_EQ(n, 0u);
}

// ---- G2CLUS-176: full F.1 surface ----

namespace irc = bolt::lakehouse::iceberg_rest;

static void make_impl(bolt::Arena* arena, irc::Impl* impl, uint8_t auth) {
    irc::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://catalog.example.com",
                 sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.prefix, "ws/main", sizeof(cfg.prefix) - 1u);
    cfg.auth_mode = auth;
    std::strncpy(cfg.oauth_token_url, "https://auth.example.com/oauth/token",
                 sizeof(cfg.oauth_token_url) - 1u);
    std::strncpy(cfg.oauth_client_id.value, "cid-123",
                 sizeof(cfg.oauth_client_id.value) - 1u);
    std::strncpy(cfg.oauth_client_secret.value, "secret-xyz",
                 sizeof(cfg.oauth_client_secret.value) - 1u);
    std::strncpy(cfg.oauth_scope, "PRINCIPAL_ROLE:ALL", sizeof(cfg.oauth_scope) - 1u);
    std::memset(impl, 0, sizeof(*impl));
    impl->cfg = cfg;
    impl->arena = arena;
}

TEST(IcebergRest, OAuth2TokenRequestShape) {
    bolt::Arena arena;
    irc::Impl impl;
    make_impl(&arena, &impl, irc::kAuthOAuth2);

    bolt::net::HttpRequest req;
    char body[1024];
    ASSERT_TRUE(irc::irc_build_oauth2_token_request(&impl, &req, body,
                                                    sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url, "https://auth.example.com/oauth/token");

    bool ct_form = false;
    for (uint32_t i = 0; i < req.header_count; ++i) {
        if (std::strcmp(req.headers[i].name, "Content-Type") == 0 &&
            std::strcmp(req.headers[i].value,
                        "application/x-www-form-urlencoded") == 0) {
            ct_form = true;
        }
    }
    EXPECT_TRUE(ct_form);
    ASSERT_GT(req.body_len, 0u);
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("grant_type=client_credentials"), std::string::npos);
    EXPECT_NE(sbody.find("client_id=cid-123"), std::string::npos);
    EXPECT_NE(sbody.find("client_secret=secret-xyz"), std::string::npos);
    EXPECT_NE(sbody.find("scope=PRINCIPAL_ROLE:ALL"), std::string::npos);
}

TEST(IcebergRest, PaginationCursorThreadedIntoUrl) {
    bolt::Arena arena;
    irc::Impl impl;
    make_impl(&arena, &impl, irc::kAuthNone);

    bolt::net::HttpRequest req;
    ASSERT_TRUE(irc::irc_build_list_namespaces_request_paged(
        &impl, "tok-PAGE-2", 50u, &req));
    EXPECT_NE(std::strstr(req.url, "/v1/ws/main/namespaces"), nullptr);
    EXPECT_NE(std::strstr(req.url, "pageToken=tok-PAGE-2"), nullptr);
    EXPECT_NE(std::strstr(req.url, "pageSize=50"), nullptr);

    // No paging args → bare URL, no '?'.
    bolt::net::HttpRequest req0;
    ASSERT_TRUE(irc::irc_build_list_namespaces_request_paged(
        &impl, nullptr, 0u, &req0));
    EXPECT_EQ(std::strchr(req0.url, '?'), nullptr);
}

TEST(IcebergRest, ParseNamespacesNextPageToken) {
    bolt::Arena arena;
    const char* body =
        "{\"namespaces\":[[\"a\"],[\"b\"]],\"next-page-token\":\"NEXT-99\"}";
    CatalogName out[8];
    uint32_t n = 0;
    char nxt[64];
    ASSERT_TRUE(irc::irc_parse_namespaces_paged(
        reinterpret_cast<const uint8_t*>(body),
        static_cast<uint32_t>(std::strlen(body)), &arena, out, 8, &n, nxt,
        sizeof(nxt)));
    EXPECT_EQ(n, 2u);
    EXPECT_STREQ(out[0].name, "a");
    EXPECT_STREQ(out[1].name, "b");
    EXPECT_STREQ(nxt, "NEXT-99");
}

TEST(IcebergRest, MultiTableCommitBody) {
    const char* t0 = "{\"identifier\":{\"namespace\":[\"db\"],\"name\":\"t0\"}}";
    const char* t1 = "{\"identifier\":{\"namespace\":[\"db\"],\"name\":\"t1\"}}";
    const char* tables[2] = { t0, t1 };
    char out[2048];
    uint32_t len = irc::irc_build_tx_commit_body(tables, 2u, out, sizeof(out));
    ASSERT_GT(len, 0u);
    std::string s(out, len);
    EXPECT_NE(s.find("\"table-changes\":["), std::string::npos);
    EXPECT_NE(s.find("\"name\":\"t0\""), std::string::npos);
    EXPECT_NE(s.find("\"name\":\"t1\""), std::string::npos);
    // Exactly one comma separating the two table objects.
    EXPECT_NE(s.find("}},{"), std::string::npos);
    EXPECT_EQ(s.back(), '}');
}

TEST(IcebergRest, SetSnapshotRefUpdateShape) {
    char out[512];
    uint32_t len = irc::irc_build_set_snapshot_ref_update(
        "release", "tag", 123456789012345LL, out, sizeof(out));
    ASSERT_GT(len, 0u);
    std::string s(out, len);
    EXPECT_NE(s.find("\"action\":\"set-snapshot-ref\""), std::string::npos);
    EXPECT_NE(s.find("\"ref-name\":\"release\""), std::string::npos);
    EXPECT_NE(s.find("\"type\":\"tag\""), std::string::npos);
    EXPECT_NE(s.find("\"snapshot-id\":123456789012345"), std::string::npos);
}

TEST(IcebergRest, ParseSnapshots) {
    bolt::Arena arena;
    const char* body =
        "{\"snapshots\":["
        "{\"snapshot-id\":11,\"sequence-number\":1,\"operation\":\"append\"},"
        "{\"snapshot-id\":22,\"parent-snapshot-id\":11,"
        "\"sequence-number\":2,\"operation\":\"overwrite\"}]}";
    irc::SnapshotRef out[8];
    uint32_t n = 0;
    ASSERT_TRUE(irc::irc_parse_snapshots(
        reinterpret_cast<const uint8_t*>(body),
        static_cast<uint32_t>(std::strlen(body)), &arena, out, 8, &n));
    ASSERT_EQ(n, 2u);
    EXPECT_EQ(out[0].snapshot_id, 11);
    EXPECT_EQ(out[0].parent_snapshot_id, -1);   // absent → -1
    EXPECT_EQ(out[0].sequence_number, 1);
    EXPECT_STREQ(out[0].operation, "append");
    EXPECT_EQ(out[1].snapshot_id, 22);
    EXPECT_EQ(out[1].parent_snapshot_id, 11);
    EXPECT_STREQ(out[1].operation, "overwrite");
}

TEST(IcebergRest, Base64StandardVectors) {
    // RFC4648 canonical vectors (the bytes a Basic header carries).
    struct { const char* in; const char* out; } v[] = {
        { "user:pass", "dXNlcjpwYXNz" },
        { "f", "Zg==" },
        { "fo", "Zm8=" },
        { "foo", "Zm9v" },
        { "foob", "Zm9vYg==" },
        { "fooba", "Zm9vYmE=" },
        { "foobar", "Zm9vYmFy" },
    };
    for (auto& c : v) {
        char out[64];
        uint32_t n = irc::irc_base64_std(
            reinterpret_cast<const uint8_t*>(c.in),
            static_cast<uint32_t>(std::strlen(c.in)), out, sizeof(out));
        ASSERT_GT(n, 0u);
        EXPECT_STREQ(out, c.out);
    }
}
