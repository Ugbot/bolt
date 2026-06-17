// Tests for the Gravitino catalog (G2CLUS-182): metalake/catalog/schema/table CRUD.

#include "bolt/lakehouse/catalog/gravitino.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

namespace grv = bolt::lakehouse::gravitino;

static bool find_header(const bolt::net::HttpRequest& req, const char* name,
                        const char* val) {
    for (uint32_t i = 0; i < req.header_count; ++i) {
        if (std::strcmp(req.headers[i].name, name) == 0 &&
            std::strcmp(req.headers[i].value, val) == 0)
            return true;
    }
    return false;
}

TEST(Gravitino, CreateMetalakeRequestShape) {
    bolt::net::HttpRequest req;
    char body[512];
    ASSERT_TRUE(grv::gravitino_build_create_metalake(
        "https://grav.example.com", "tok-G", "analytics", "the lake", &req,
        body, sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url, "https://grav.example.com/api/metalakes");
    EXPECT_TRUE(find_header(req, "Authorization", "Bearer tok-G"));
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"name\":\"analytics\""), std::string::npos);
    EXPECT_NE(sbody.find("\"comment\":\"the lake\""), std::string::npos);
}

TEST(Gravitino, CreateCatalogRequestShape) {
    bolt::net::HttpRequest req;
    char body[512];
    ASSERT_TRUE(grv::gravitino_build_create_catalog(
        "https://grav.example.com", "tok-G", "analytics", "warehouse",
        "RELATIONAL", "lakehouse-iceberg", &req, body, sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url,
                 "https://grav.example.com/api/metalakes/analytics/catalogs");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"name\":\"warehouse\""), std::string::npos);
    EXPECT_NE(sbody.find("\"type\":\"RELATIONAL\""), std::string::npos);
    EXPECT_NE(sbody.find("\"provider\":\"lakehouse-iceberg\""), std::string::npos);
}

TEST(Gravitino, CreateSchemaRequestShape) {
    bolt::net::HttpRequest req;
    char body[512];
    ASSERT_TRUE(grv::gravitino_build_create_schema(
        "https://grav.example.com", "tok-G", "analytics", "warehouse", "sales",
        "sales schema", &req, body, sizeof(body)));
    EXPECT_STREQ(req.url,
                 "https://grav.example.com/api/metalakes/analytics/catalogs/"
                 "warehouse/schemas");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"name\":\"sales\""), std::string::npos);
}

TEST(Gravitino, CreateTableRequestShape) {
    bolt::net::HttpRequest req;
    char body[512];
    ASSERT_TRUE(grv::gravitino_build_create_table(
        "https://grav.example.com", "tok-G", "analytics", "warehouse", "sales",
        "orders", &req, body, sizeof(body)));
    EXPECT_STREQ(req.url,
                 "https://grav.example.com/api/metalakes/analytics/catalogs/"
                 "warehouse/schemas/sales/tables");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"name\":\"orders\""), std::string::npos);
    EXPECT_NE(sbody.find("\"columns\":[]"), std::string::npos);
}

TEST(Gravitino, OpenBindsCatalog) {
    bolt::Arena arena;
    grv::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://grav.example.com",
                 sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.metalake, "analytics", sizeof(cfg.metalake) - 1u);
    bolt::lakehouse::Catalog cat;
    ASSERT_TRUE(grv::open(&cat, &arena, &cfg));
    ASSERT_NE(cat.vt, nullptr);
    ASSERT_NE(cat.impl, nullptr);
}
