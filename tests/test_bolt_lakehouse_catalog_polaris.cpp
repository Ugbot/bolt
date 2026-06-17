// Tests for the Polaris catalog (G2CLUS-180): Iceberg REST reuse + management API.

#include "bolt/lakehouse/catalog/polaris.h"
#include "bolt/lakehouse/catalog/iceberg_rest.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

namespace pol = bolt::lakehouse::polaris;
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

// open() must wire the iceberg_rest client with the /api/catalog/v1 prefix and
// OAuth2 client-credentials. We verify by shaping a list-namespaces request off
// the resulting Impl and checking the URL prefix + OAuth body shape.
TEST(Polaris, OpenWiresIcebergRestPrefixAndOAuth) {
    bolt::Arena arena;
    pol::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.base_url, "https://polaris.example.com",
                 sizeof(cfg.base_url) - 1u);
    std::strncpy(cfg.oauth_token_url,
                 "https://polaris.example.com/api/catalog/v1/oauth/tokens",
                 sizeof(cfg.oauth_token_url) - 1u);
    std::strncpy(cfg.oauth_client_id.value, "polaris-cid",
                 sizeof(cfg.oauth_client_id.value) - 1u);
    std::strncpy(cfg.oauth_client_secret.value, "polaris-sec",
                 sizeof(cfg.oauth_client_secret.value) - 1u);
    std::strncpy(cfg.oauth_scope, "PRINCIPAL_ROLE:ALL", sizeof(cfg.oauth_scope) - 1u);

    bolt::lakehouse::Catalog cat;
    ASSERT_TRUE(pol::open(&cat, &arena, &cfg));
    irc::Impl* impl = static_cast<irc::Impl*>(cat.impl);
    ASSERT_EQ(impl->cfg.auth_mode, irc::kAuthOAuth2);
    EXPECT_STREQ(impl->cfg.prefix, "api/catalog/v1");

    bolt::net::HttpRequest req;
    ASSERT_TRUE(irc::irc_build_list_namespaces_request_paged(impl, nullptr, 0u,
                                                             &req));
    EXPECT_NE(std::strstr(req.url, "/v1/api/catalog/v1/namespaces"), nullptr);

    // OAuth2 token body carries the client-credentials grant + Polaris scope.
    bolt::net::HttpRequest treq;
    char body[1024];
    ASSERT_TRUE(irc::irc_build_oauth2_token_request(impl, &treq, body,
                                                    sizeof(body)));
    std::string sbody(reinterpret_cast<const char*>(treq.body), treq.body_len);
    EXPECT_NE(sbody.find("grant_type=client_credentials"), std::string::npos);
    EXPECT_NE(sbody.find("client_id=polaris-cid"), std::string::npos);
    EXPECT_NE(sbody.find("scope=PRINCIPAL_ROLE:ALL"), std::string::npos);
}

TEST(Polaris, CreatePrincipalRoleRequestShape) {
    bolt::net::HttpRequest req;
    char body[256];
    ASSERT_TRUE(pol::polaris_build_create_principal_role(
        "https://polaris.example.com", "tok-XYZ", "data_engineer", &req, body,
        sizeof(body)));
    EXPECT_STREQ(req.method, "POST");
    EXPECT_STREQ(req.url,
                 "https://polaris.example.com/api/management/v1/principal-roles");
    EXPECT_TRUE(find_header(req, "Authorization", "Bearer tok-XYZ"));
    EXPECT_TRUE(find_header(req, "Content-Type", "application/json"));
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"principalRole\""), std::string::npos);
    EXPECT_NE(sbody.find("\"name\":\"data_engineer\""), std::string::npos);
}

TEST(Polaris, GrantCatalogRoleRequestShape) {
    bolt::net::HttpRequest req;
    char body[256];
    ASSERT_TRUE(pol::polaris_build_grant_catalog_role(
        "https://polaris.example.com", "tok-XYZ", "data_engineer", "prod_cat",
        "reader", &req, body, sizeof(body)));
    EXPECT_STREQ(req.method, "PUT");
    EXPECT_STREQ(req.url,
                 "https://polaris.example.com/api/management/v1/principal-roles/"
                 "data_engineer/catalog-roles/prod_cat");
    EXPECT_TRUE(find_header(req, "Authorization", "Bearer tok-XYZ"));
    std::string sbody(reinterpret_cast<const char*>(req.body), req.body_len);
    EXPECT_NE(sbody.find("\"catalogRole\""), std::string::npos);
    EXPECT_NE(sbody.find("\"name\":\"reader\""), std::string::npos);
}
