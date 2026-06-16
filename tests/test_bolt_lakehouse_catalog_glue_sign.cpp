// Test that Glue requests are SigV4-signed with the expected header shape.

#include "bolt/lakehouse/catalog/glue.h"

#include <gtest/gtest.h>

#include <cstring>

using bolt::lakehouse::glue::Config;
using bolt::lakehouse::glue::Impl;

TEST(GlueSign, AuthorizationHeaderShape) {
    bolt::Arena arena;
    Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.region, "us-east-1", sizeof(cfg.region) - 1u);
    std::strncpy(cfg.access_key.value, "AKIAIOSFODNN7EXAMPLE",
                 sizeof(cfg.access_key.value) - 1u);
    std::strncpy(cfg.secret_key.value,
                 "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                 sizeof(cfg.secret_key.value) - 1u);

    Impl impl;
    std::memset(&impl, 0, sizeof(impl));
    impl.cfg = cfg;
    impl.arena = &arena;

    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    std::strncpy(req.method, "POST", sizeof(req.method) - 1u);
    std::strncpy(req.url, "https://glue.us-east-1.amazonaws.com/",
                 sizeof(req.url) - 1u);
    const char* body = "{}";
    req.body = reinterpret_cast<const uint8_t*>(body);
    req.body_len = 2;
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/x-amz-json-1.1");
    bolt::net::http_request_add_header(&req, "X-Amz-Target", "AWSGlue.GetDatabases");

    ASSERT_TRUE(bolt::lakehouse::glue::glue_sign_request_with_date(
        &impl, &req, "20260617T120000Z"));

    const char* auth = nullptr;
    const char* date = nullptr;
    const char* cs = nullptr;
    for (uint32_t i = 0; i < req.header_count; ++i) {
        if (std::strcmp(req.headers[i].name, "Authorization") == 0)
            auth = req.headers[i].value;
        if (std::strcmp(req.headers[i].name, "X-Amz-Date") == 0)
            date = req.headers[i].value;
        if (std::strcmp(req.headers[i].name, "X-Amz-Content-Sha256") == 0)
            cs = req.headers[i].value;
    }
    ASSERT_NE(auth, nullptr);
    EXPECT_STREQ(date, "20260617T120000Z");
    ASSERT_NE(cs, nullptr);
    // Authorization: AWS4-HMAC-SHA256 Credential=AKIA.../20260617/us-east-1/glue/aws4_request, ...
    EXPECT_EQ(std::strncmp(auth, "AWS4-HMAC-SHA256 ", 17), 0);
    EXPECT_NE(std::strstr(auth, "Credential=AKIAIOSFODNN7EXAMPLE/20260617"),
              nullptr);
    EXPECT_NE(std::strstr(auth, "us-east-1/glue/aws4_request"), nullptr);
    EXPECT_NE(std::strstr(auth, "SignedHeaders="), nullptr);
    EXPECT_NE(std::strstr(auth, "Signature="), nullptr);
}
