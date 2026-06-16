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
