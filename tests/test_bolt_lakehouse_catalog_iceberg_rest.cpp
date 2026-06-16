// Tests for the Iceberg REST catalog wire shape + JSON parse.

#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <gtest/gtest.h>

#include <cstring>

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
