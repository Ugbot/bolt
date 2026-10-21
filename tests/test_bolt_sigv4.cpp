// AWS SigV4 — software SHA-256 / HMAC-SHA256 + the GET-vanilla signing vector.

#include "bolt/crypto/sigv4.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace {

using namespace bolt::crypto;

// SHA-256("") is the well-known empty digest.
TEST(BoltSigV4, Sha256Empty) {
    char hex[65];
    sha256_empty_hex(hex);
    EXPECT_STREQ(hex,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

// SHA-256("abc") = ba7816bf...
TEST(BoltSigV4, Sha256Abc) {
    char hex[65];
    sha256_hex(reinterpret_cast<const uint8_t*>("abc"), 3, hex);
    EXPECT_STREQ(hex,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// RFC 4231 HMAC-SHA256 test case 1: key=0x0b*20, data="Hi There".
TEST(BoltSigV4, HmacRfc4231Case1) {
    uint8_t key[20];
    std::memset(key, 0x0b, sizeof(key));
    uint8_t mac[32];
    hmac_sha256(key, 20, reinterpret_cast<const uint8_t*>("Hi There"), 8, mac);
    static const char* kHex = "0123456789abcdef";
    char out[65];
    for (int i = 0; i < 32; ++i) {
        out[2*i] = kHex[(mac[i] >> 4) & 0xF];
        out[2*i+1] = kHex[mac[i] & 0xF];
    }
    out[64] = '\0';
    EXPECT_STREQ(out,
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

// The canonical AWS "get-vanilla" SigV4 test vector.
TEST(BoltSigV4, GetVanilla) {
    char empty_hash[65];
    sha256_empty_hex(empty_hash);

    SigV4Request req;
    req.method = "GET";
    req.host = "example.amazonaws.com";
    req.path = "/";
    req.query_string = "";
    req.service = "service";
    req.region = "us-east-1";
    req.amz_date = "20150830T123600Z";
    req.payload_sha256 = empty_hash;
    req.access_key = "AKIDEXAMPLE";
    req.secret_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    req.session_token = nullptr;

    char auth[kSigV4MaxAuth];
    char date_out[24];
    SigV4Result out;
    out.auth_header = auth;
    out.auth_cap = sizeof(auth);
    out.amz_date_out = date_out;
    out.amz_date_cap = sizeof(date_out);

    ASSERT_TRUE(sigv4_sign(&req, &out));

    // Credential scope + algorithm prefix must match exactly.
    const std::string a(auth);
    EXPECT_EQ(a.rfind("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/"
                      "us-east-1/service/aws4_request", 0), 0u)
        << "auth = " << a;
    EXPECT_NE(a.find("SignedHeaders=host;x-amz-content-sha256;x-amz-date"),
              std::string::npos);
    EXPECT_NE(a.find("Signature="), std::string::npos);
    // Exact signature for this (S3-style, x-amz-content-sha256-included)
    // canonical request — cross-checked against a reference HMAC chain.
    EXPECT_NE(a.find("Signature=726c5c4879a6b4ccbbd3b24edbd6b8826d34f8745"
                     "0fbbf4e85546fc7ba9c1642"), std::string::npos);
    EXPECT_STREQ(date_out, "20150830T123600Z");
}

// Session token adds the x-amz-security-token signed header.
TEST(BoltSigV4, SessionTokenSignedHeader) {
    char empty_hash[65];
    sha256_empty_hex(empty_hash);
    SigV4Request req;
    req.method = "GET"; req.host = "example.amazonaws.com"; req.path = "/";
    req.query_string = ""; req.service = "service"; req.region = "us-east-1";
    req.amz_date = "20150830T123600Z"; req.payload_sha256 = empty_hash;
    req.access_key = "AKIDEXAMPLE";
    req.secret_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    req.session_token = "SESSIONTOKEN";

    char auth[kSigV4MaxAuth];
    char date_out[24];
    SigV4Result out{auth, sizeof(auth), date_out, sizeof(date_out)};
    ASSERT_TRUE(sigv4_sign(&req, &out));
    EXPECT_NE(std::string(auth).find("x-amz-security-token"),
              std::string::npos);
}

// GCS XML API create-only PUT: the signed x-goog-if-generation-match header.
// Signature from an independent Python hmac/hashlib SigV4 chain.
TEST(BoltSigV4, ExtraSignedHeaderGcsGenerationMatch) {
    char empty_hash[65];
    sha256_empty_hex(empty_hash);
    SigV4Request req;
    req.method = "PUT"; req.host = "storage.googleapis.com";
    req.path = "/bkt/d/_delta_log/0.json"; req.query_string = "";
    req.service = "s3"; req.region = "auto";
    req.amz_date = "20260924T120000Z"; req.payload_sha256 = empty_hash;
    req.access_key = "GOOG1EXAMPLE"; req.secret_key = "GOOGSECRETEXAMPLE";
    req.session_token = nullptr;
    req.extra_header = "x-goog-if-generation-match:0";
    char auth[kSigV4MaxAuth];
    char date_out[24];
    SigV4Result out{auth, sizeof(auth), date_out, sizeof(date_out)};
    ASSERT_TRUE(sigv4_sign(&req, &out));
    const std::string a(auth);
    EXPECT_NE(a.find("SignedHeaders=host;x-amz-content-sha256;x-amz-date;"
                     "x-goog-if-generation-match,"), std::string::npos) << a;
    EXPECT_NE(a.find("Signature=729f009732e3e6f817da0f3b3df601bbdbda8d9e9"
                     "2ed30051a86386743fc8f79"), std::string::npos) << a;
}

// An extra header must be lowercase, well-formed and sort last.
TEST(BoltSigV4, ExtraHeaderRejectsBadNames) {
    char empty_hash[65];
    sha256_empty_hex(empty_hash);
    SigV4Request req;
    req.method = "PUT"; req.host = "h"; req.path = "/"; req.query_string = "";
    req.service = "s3"; req.region = "auto";
    req.amz_date = "20260924T120000Z"; req.payload_sha256 = empty_hash;
    req.access_key = "a"; req.secret_key = "s"; req.session_token = nullptr;
    char auth[kSigV4MaxAuth];
    char date_out[24];
    SigV4Result out{auth, sizeof(auth), date_out, sizeof(date_out)};
    for (const char* bad : {"X-Goog-If-Generation-Match:0", "no-colon",
                            ":0", "content-type:x", "x-goog a:0"}) {
        req.extra_header = bad;
        EXPECT_FALSE(sigv4_sign(&req, &out)) << bad;
    }
}

// Null required fields must fail cleanly.
TEST(BoltSigV4, RejectsNullFields) {
    SigV4Request req;
    std::memset(&req, 0, sizeof(req));
    char auth[kSigV4MaxAuth];
    char date_out[24];
    SigV4Result out{auth, sizeof(auth), date_out, sizeof(date_out)};
    EXPECT_FALSE(sigv4_sign(&req, &out));
}

}  // namespace

namespace {
std::string independent_if_match_authorization() {
    using namespace bolt::crypto;
    char empty[65]{}, canonical_hash[65]{};
    sha256_empty_hex(empty);
    const std::string headers = "host;if-match;x-amz-content-sha256;x-amz-date;x-amz-security-token;x-goog-if-generation-match";
    const std::string canonical = std::string("GET\n/object\n\nhost:example.com\nif-match:\"opaque-test\"\nx-amz-content-sha256:") +
        empty + "\nx-amz-date:20261021T120000Z\nx-amz-security-token:session-token\nx-goog-if-generation-match:0\n\n" + headers + "\n" + empty;
    sha256_hex(reinterpret_cast<const uint8_t*>(canonical.data()), canonical.size(), canonical_hash);
    const std::string scope = "20261021/us-east-1/s3/aws4_request";
    const std::string to_sign = std::string("AWS4-HMAC-SHA256\n20261021T120000Z\n") + scope + "\n" + canonical_hash;
    uint8_t date_key[32]{}, region_key[32]{}, service_key[32]{}, signing_key[32]{}, signature[32]{};
    const auto* secret = reinterpret_cast<const uint8_t*>("AWS4secret");
    hmac_sha256(secret, 10, reinterpret_cast<const uint8_t*>("20261021"), 8, date_key);
    hmac_sha256(date_key, 32, reinterpret_cast<const uint8_t*>("us-east-1"), 9, region_key);
    hmac_sha256(region_key, 32, reinterpret_cast<const uint8_t*>("s3"), 2, service_key);
    hmac_sha256(service_key, 32, reinterpret_cast<const uint8_t*>("aws4_request"), 12, signing_key);
    hmac_sha256(signing_key, 32, reinterpret_cast<const uint8_t*>(to_sign.data()), to_sign.size(), signature);
    constexpr char hex[] = "0123456789abcdef";
    char signature_hex[65]{};
    for (uint32_t i = 0; i < 32; ++i) {
        signature_hex[i * 2u] = hex[signature[i] >> 4u];
        signature_hex[i * 2u + 1u] = hex[signature[i] & 15u];
    }
    return "AWS4-HMAC-SHA256 Credential=access/" + scope + ", SignedHeaders=" + headers + ", Signature=" + signature_hex;
}

TEST(BoltSigV4, IfMatchHasIndependentLexicalCanonicalOrderWithSessionAndExtra) {
    char empty[65]{};
    bolt::crypto::sha256_empty_hex(empty);
    bolt::crypto::SigV4Request request{};
    request.method = "GET"; request.host = "example.com"; request.path = "/object";
    request.query_string = ""; request.service = "s3"; request.region = "us-east-1";
    request.amz_date = "20261021T120000Z"; request.payload_sha256 = empty;
    request.access_key = "access"; request.secret_key = "secret";
    request.session_token = "session-token"; request.extra_header = "x-goog-if-generation-match:0";
    request.if_match = "\"opaque-test\"";
    char auth[bolt::crypto::kSigV4MaxAuth]{}, date[24]{};
    bolt::crypto::SigV4Result out{auth, sizeof(auth), date, sizeof(date)};
    ASSERT_TRUE(bolt::crypto::sigv4_sign(&request, &out));
    EXPECT_EQ(std::string(auth), independent_if_match_authorization());
    EXPECT_STREQ(date, request.amz_date);
    for (const char* invalid : {"W/\"weak\"", "*", "\"one\",\"two\"", "\"space value\"", "\"tag\r\nInjected:yes\""}) {
        request.if_match = invalid;
        EXPECT_FALSE(bolt::crypto::sigv4_sign(&request, &out)) << invalid;
    }
}
}  // namespace
