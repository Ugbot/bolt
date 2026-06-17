// Tests for the Nessie catalog (G2CLUS-181): Iceberg REST reuse + native v2 trees.

#include "bolt/lakehouse/catalog/nessie.h"
#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

namespace nes = bolt::lakehouse::nessie;
namespace irc = bolt::lakehouse::iceberg_rest;

static bool find_header(const bolt::net::HttpRequest& req, const char* name,
                        const char* val) {
    for (uint32_t i = 0; i < req.header_count; ++i) {
        if (std::strcmp(req.headers[i].name, name) == 0 &&
            std::strcmp(req.headers[i].value, val) == 0)
            return true;
    }
    return false;
}

TEST(Nessie, OpenWiresIcebergRestPrefix) {
    bolt::Arena arena;
    nes::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://nessie.example.com",
                 sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.bearer_token.value, "tok-N", sizeof(cfg.bearer_token.value) - 1u);
    std::strncpy(cfg.reference, "main", sizeof(cfg.reference) - 1u);

    bolt::lakehouse::Catalog cat;
    ASSERT_TRUE(nes::open(&cat, &arena, &cfg));
    irc::Impl* impl = static_cast<irc::Impl*>(cat.impl);
    EXPECT_STREQ(impl->cfg.prefix, "iceberg");
    EXPECT_EQ(impl->cfg.auth_mode, irc::kAuthBearer);
}

TEST(Nessie, GetReferenceRequestShape) {
    bolt::net::HttpRequest req;
    ASSERT_TRUE(nes::nessie_build_get_reference(
        "https://nessie.example.com", "tok-N", "main", &req));
    EXPECT_STREQ(req.method, "GET");
    EXPECT_STREQ(req.url, "https://nessie.example.com/api/v2/trees/main");
    EXPECT_TRUE(find_header(req, "Authorization", "Bearer tok-N"));
}

TEST(Nessie, CommitBodyShape) {
    bolt::net::HttpRequest req;
    char body[2048];
    ASSERT_TRUE(nes::nessie_build_commit(
        "https://nessie.example.com", "tok-N", "dev", "abc123",
        "add table foo",
        "[{\"type\":\"PUT\",\"key\":{\"elements\":[\"ns\",\"foo\"]}}]", &req,
        body, sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url,
                 "https://nessie.example.com/api/v2/trees/dev@abc123/history/commit");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"commitMeta\""), std::string::npos);
    EXPECT_NE(sbody.find("\"message\":\"add table foo\""), std::string::npos);
    EXPECT_NE(sbody.find("\"operations\":[{\"type\":\"PUT\""), std::string::npos);
}

TEST(Nessie, CommitWithoutExpectedHashOmitsAt) {
    bolt::net::HttpRequest req;
    char body[1024];
    ASSERT_TRUE(nes::nessie_build_commit("https://h", "", "main", "", "msg",
                                         "[]", &req, body, sizeof(body)));
    EXPECT_STREQ(req.url, "https://h/api/v2/trees/main/history/commit");
    EXPECT_EQ(std::strchr(req.url, '@'), nullptr);
}

TEST(Nessie, MergeBodyShape) {
    bolt::net::HttpRequest req;
    char body[512];
    ASSERT_TRUE(nes::nessie_build_merge("https://h", "tok", "feature", "ff00",
                                        "main", &req, body, sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url, "https://h/api/v2/trees/main/history/merge");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"fromRefName\":\"feature\""), std::string::npos);
    EXPECT_NE(sbody.find("\"fromHash\":\"ff00\""), std::string::npos);
}

TEST(Nessie, TransplantBodyShape) {
    bolt::net::HttpRequest req;
    char body[1024];
    const char* hashes[2] = { "h1", "h2" };
    ASSERT_TRUE(nes::nessie_build_transplant("https://h", "tok", "src", "main",
                                             hashes, 2u, &req, body,
                                             sizeof(body)));
    EXPECT_STREQ(req.url, "https://h/api/v2/trees/main/history/transplant");
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"fromRefName\":\"src\""), std::string::npos);
    EXPECT_NE(sbody.find("\"hashesToTransplant\":[\"h1\",\"h2\"]"),
              std::string::npos);
}

TEST(Nessie, CompareReferencesRequestShape) {
    bolt::net::HttpRequest req;
    ASSERT_TRUE(nes::nessie_build_compare_references("https://h", "tok", "main",
                                                     "dev", &req));
    EXPECT_STREQ(req.method, "GET");
    EXPECT_STREQ(req.url, "https://h/api/v2/trees/main/diff/dev");
}
