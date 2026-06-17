// Tests for the Databricks Unity Catalog REST client wire shape:
// PAT/Bearer header injection, 3-level full_name encoding, table-create body
// (data_source_format + storage_location), a GRANT (permissions PATCH) body,
// and a Delta Sharing share-create body.

#include "bolt/lakehouse/catalog/unity.h"

#include <gtest/gtest.h>

#include <cstring>

using bolt::lakehouse::unity::Config;
using bolt::lakehouse::unity::DataFormat;
using bolt::lakehouse::unity::Impl;
using bolt::lakehouse::unity::Privilege;
using bolt::lakehouse::unity::Securable;

namespace {

Impl make_impl(bolt::Arena* arena, const char* token) {
    Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://dbc-x.cloud.databricks.com",
                 sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.bearer_token.value, token,
                 sizeof(cfg.bearer_token.value) - 1u);
    Impl impl;
    std::memset(&impl, 0, sizeof(impl));
    impl.cfg = cfg;
    impl.arena = arena;
    return impl;
}

const char* header_value(const bolt::net::HttpRequest& req, const char* name) {
    for (uint32_t i = 0; i < req.header_count; ++i)
        if (std::strcmp(req.headers[i].name, name) == 0)
            return req.headers[i].value;
    return nullptr;
}

bool body_contains(const bolt::net::HttpRequest& req, const char* needle) {
    if (req.body == nullptr) return false;
    return std::strstr(reinterpret_cast<const char*>(req.body), needle)
           != nullptr;
}

}  // namespace

TEST(UnityCatalog, FullNameEncoding) {
    char fn[256];
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_full_name(
        "main", "sales", "orders", fn, sizeof(fn)));
    EXPECT_STREQ(fn, "main.sales.orders");

    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_full_name(
        "main", "sales", nullptr, fn, sizeof(fn)));
    EXPECT_STREQ(fn, "main.sales");

    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_full_name(
        "main", "", "", fn, sizeof(fn)));
    EXPECT_STREQ(fn, "main");

    // All-empty fails; a too-small buffer fails (no truncation).
    EXPECT_FALSE(bolt::lakehouse::unity::uc_build_full_name(
        nullptr, nullptr, nullptr, fn, sizeof(fn)));
    char tiny[4];
    EXPECT_FALSE(bolt::lakehouse::unity::uc_build_full_name(
        "main", "sales", "orders", tiny, sizeof(tiny)));
}

TEST(UnityCatalog, PatHeaderInjection) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "dapi-secret-pat");
    bolt::net::HttpRequest req;
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_list_catalogs_request(
        &impl, &req));
    EXPECT_STREQ(req.method, "GET");
    EXPECT_NE(std::strstr(req.url, "/api/2.1/unity-catalog/catalogs"), nullptr);
    const char* auth = header_value(req, "Authorization");
    ASSERT_NE(auth, nullptr);
    EXPECT_STREQ(auth, "Bearer dapi-secret-pat");
}

TEST(UnityCatalog, GetTableUsesFullNameInPath) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_get_table_request(
        &impl, "main.sales.orders", &req));
    EXPECT_STREQ(req.method, "GET");
    EXPECT_NE(std::strstr(req.url, "/tables/main.sales.orders"), nullptr);
}

TEST(UnityCatalog, CreateTableBody) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    char body[768];
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_create_table_request(
        &impl, "main", "sales", "orders", DataFormat::kIceberg,
        "s3://bucket/orders", body, sizeof(body), &req));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_NE(std::strstr(req.url, "/tables"), nullptr);
    EXPECT_TRUE(body_contains(req, "\"data_source_format\":\"ICEBERG\""));
    EXPECT_TRUE(body_contains(req, "\"storage_location\":\"s3://bucket/orders\""));
    EXPECT_TRUE(body_contains(req, "\"catalog_name\":\"main\""));
    EXPECT_TRUE(body_contains(req, "\"schema_name\":\"sales\""));
    // Content-Type set for a body-bearing request.
    EXPECT_STREQ(header_value(req, "Content-Type"), "application/json");
}

TEST(UnityCatalog, GrantBody) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    char body[512];
    Privilege privs[2] = {Privilege::kSelect, Privilege::kModify};
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_grant_request(
        &impl, Securable::kTable, "main.sales.orders", "alice@corp.com",
        privs, 2u, body, sizeof(body), &req));
    EXPECT_STREQ(req.method, "PATCH");
    EXPECT_NE(std::strstr(req.url, "/permissions/table/main.sales.orders"),
              nullptr);
    EXPECT_TRUE(body_contains(req, "\"principal\":\"alice@corp.com\""));
    EXPECT_TRUE(body_contains(req, "\"SELECT\""));
    EXPECT_TRUE(body_contains(req, "\"MODIFY\""));
    EXPECT_TRUE(body_contains(req, "\"add\":[\"SELECT\",\"MODIFY\"]"));
}

TEST(UnityCatalog, ShareCreateBody) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    char body[256];
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_create_share_request(
        &impl, "q1_share", "Q1 datasets", body, sizeof(body), &req));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_NE(std::strstr(req.url, "/shares"), nullptr);
    EXPECT_TRUE(body_contains(req, "\"name\":\"q1_share\""));
    EXPECT_TRUE(body_contains(req, "\"comment\":\"Q1 datasets\""));
}

TEST(UnityCatalog, ListSchemasQueryParam) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_list_schemas_request(
        &impl, "main", &req));
    EXPECT_NE(std::strstr(req.url, "/schemas?catalog_name=main"), nullptr);
}

TEST(UnityCatalog, ColumnLineageRequest) {
    bolt::Arena arena;
    Impl impl = make_impl(&arena, "tok");
    bolt::net::HttpRequest req;
    ASSERT_TRUE(bolt::lakehouse::unity::uc_build_column_lineage_request(
        &impl, "main.sales.orders", "amount", &req));
    EXPECT_NE(std::strstr(req.url,
              "/lineage-tracking/column-lineage?table_name=main.sales.orders"
              "&column_name=amount"), nullptr);
}
