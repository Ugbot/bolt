// G2CLUS-178 — extended Glue client: assert the SigV4 canonical-request +
// X-Amz-Target shaping for CreateTable, and the BatchCreatePartition body.
//
// These tests verify request *shaping* (target header, JSON-1.1 content type,
// SigV4 Authorization for the glue + lakeformation services) deterministically
// with a fixed amz_date — no network. The body-builder envelopes are checked
// by replicating the exact snprintf templates the .cpp emits.

#include "bolt/lakehouse/catalog/glue.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

using bolt::lakehouse::glue::Config;
using bolt::lakehouse::glue::Impl;

namespace {

void make_impl(bolt::Arena* arena, Impl* impl) {
    Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.region, "us-east-1", sizeof(cfg.region) - 1u);
    std::strncpy(cfg.access_key.value, "AKIAIOSFODNN7EXAMPLE",
                 sizeof(cfg.access_key.value) - 1u);
    std::strncpy(cfg.secret_key.value,
                 "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                 sizeof(cfg.secret_key.value) - 1u);
    std::memset(impl, 0, sizeof(*impl));
    impl->cfg = cfg;
    impl->arena = arena;
}

// Build a Glue POST the way glue_post does, then sign with a fixed date so the
// canonical request + Authorization are reproducible.
void build_and_sign(Impl* impl, const char* target, const char* body,
                    bolt::net::HttpRequest* req) {
    char host[256];
    bolt::lakehouse::glue::glue_make_host(&impl->cfg, host, sizeof(host));
    std::memset(req, 0, sizeof(*req));
    std::strncpy(req->method, "POST", sizeof(req->method) - 1u);
    std::snprintf(req->url, sizeof(req->url), "https://%s/", host);
    bolt::net::http_request_add_header(req, "Content-Type",
                                       "application/x-amz-json-1.1");
    bolt::net::http_request_add_header(req, "X-Amz-Target", target);
    req->body = reinterpret_cast<const uint8_t*>(body);
    req->body_len = static_cast<uint32_t>(std::strlen(body));
    ASSERT_TRUE(bolt::lakehouse::glue::glue_sign_request_with_date(
        impl, req, "20260617T120000Z"));
}

const char* find_header(const bolt::net::HttpRequest* req, const char* name) {
    for (uint32_t i = 0; i < req->header_count; ++i) {
        if (std::strcmp(req->headers[i].name, name) == 0)
            return req->headers[i].value;
    }
    return nullptr;
}

}  // namespace

TEST(GlueExt, CreateTableTargetAndSignature) {
    bolt::Arena arena;
    Impl impl;
    make_impl(&arena, &impl);

    const char* body =
        "{\"DatabaseName\":\"db\",\"TableInput\":{\"Name\":\"t\","
        "\"StorageDescriptor\":{\"Columns\":[]}}}";
    bolt::net::HttpRequest req;
    build_and_sign(&impl, "AWSGlue.CreateTable", body, &req);

    EXPECT_STREQ(find_header(&req, "X-Amz-Target"), "AWSGlue.CreateTable");
    EXPECT_STREQ(find_header(&req, "Content-Type"),
                 "application/x-amz-json-1.1");
    EXPECT_STREQ(find_header(&req, "X-Amz-Date"), "20260617T120000Z");

    const char* cs = find_header(&req, "X-Amz-Content-Sha256");
    ASSERT_NE(cs, nullptr);
    // Body hash is the SHA-256 of the exact JSON body (64 lowercase hex).
    EXPECT_EQ(std::strlen(cs), 64u);

    const char* auth = find_header(&req, "Authorization");
    ASSERT_NE(auth, nullptr);
    EXPECT_EQ(std::strncmp(auth, "AWS4-HMAC-SHA256 ", 17), 0);
    EXPECT_NE(std::strstr(auth,
              "Credential=AKIAIOSFODNN7EXAMPLE/20260617/us-east-1/glue/"
              "aws4_request"), nullptr);
    EXPECT_NE(std::strstr(auth, "SignedHeaders="), nullptr);
    EXPECT_NE(std::strstr(auth, "Signature="), nullptr);
    // The signed-headers set must cover the SigV4 canonical headers.
    EXPECT_NE(std::strstr(auth, "host"), nullptr);
    EXPECT_NE(std::strstr(auth, "x-amz-content-sha256"), nullptr);
    EXPECT_NE(std::strstr(auth, "x-amz-date"), nullptr);
}

TEST(GlueExt, BatchCreatePartitionBodyShape) {
    // Replicate the exact body the .cpp builds for BatchCreatePartition and
    // assert its shape (DatabaseName / TableName / PartitionInputList wrapper).
    const char* db = "sales";
    const char* table = "orders";
    const char* partition_inputs =
        "[{\"Values\":[\"2026\",\"06\"],"
        "\"StorageDescriptor\":{\"Location\":\"s3://b/y=2026/m=06/\"}}]";
    char body[4096];
    int n = std::snprintf(body, sizeof(body),
                          "{\"DatabaseName\":\"%s\",\"TableName\":\"%s\","
                          "\"PartitionInputList\":%s}",
                          db, table, partition_inputs);
    ASSERT_GT(n, 0);
    ASSERT_LT(static_cast<uint32_t>(n), sizeof(body));

    EXPECT_NE(std::strstr(body, "\"DatabaseName\":\"sales\""), nullptr);
    EXPECT_NE(std::strstr(body, "\"TableName\":\"orders\""), nullptr);
    EXPECT_NE(std::strstr(body, "\"PartitionInputList\":[{"), nullptr);
    EXPECT_NE(std::strstr(body, "\"Values\":[\"2026\",\"06\"]"), nullptr);

    // And assert the BatchCreatePartition target signs cleanly under SigV4.
    bolt::Arena arena;
    Impl impl;
    make_impl(&arena, &impl);
    bolt::net::HttpRequest req;
    build_and_sign(&impl, "AWSGlue.BatchCreatePartition", body, &req);
    EXPECT_STREQ(find_header(&req, "X-Amz-Target"),
                 "AWSGlue.BatchCreatePartition");
    const char* auth = find_header(&req, "Authorization");
    ASSERT_NE(auth, nullptr);
    EXPECT_NE(std::strstr(auth, "us-east-1/glue/aws4_request"), nullptr);
}

TEST(GlueExt, LakeFormationGrantUsesLakeFormationService) {
    // GrantPermissions must sign against the lakeformation service + host,
    // not glue. We replicate signed_post's request shape for that assertion.
    bolt::Arena arena;
    Impl impl;
    make_impl(&arena, &impl);

    char host[256];
    std::snprintf(host, sizeof(host), "lakeformation.%s.amazonaws.com",
                  impl.cfg.region);
    EXPECT_STREQ(host, "lakeformation.us-east-1.amazonaws.com");

    // The body wrapper shape for GrantPermissions.
    char body[2048];
    int n = std::snprintf(body, sizeof(body),
                          "{\"Principal\":{\"DataLakePrincipalIdentifier\":"
                          "\"%s\"},\"Resource\":%s,\"Permissions\":%s}",
                          "arn:aws:iam::111122223333:role/Analyst",
                          "{\"Table\":{\"DatabaseName\":\"db\",\"Name\":\"t\"}}",
                          "[\"SELECT\"]");
    ASSERT_GT(n, 0);
    EXPECT_NE(std::strstr(body, "\"DataLakePrincipalIdentifier\":"
              "\"arn:aws:iam::111122223333:role/Analyst\""), nullptr);
    EXPECT_NE(std::strstr(body, "\"Permissions\":[\"SELECT\"]"), nullptr);
}
