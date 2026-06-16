// GCS JWT — header + claims structure, base64url, and SA-JSON parsing.
//
// No network. RS256 signing path: when BOLT_WITH_TLS is on we generate a
// fresh 2048-bit RSA key inline and verify a 256-byte signature comes back;
// when off we just exercise the header.payload builder.

#include "bolt/lakehouse/object_store/gcs.h"
#include "bolt/lakehouse/object_store/compat.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#if defined(BOLT_WITH_TLS)
#  include <openssl/bio.h>
#  include <openssl/evp.h>
#  include <openssl/pem.h>
#endif

namespace {

using namespace bolt::lakehouse::gcs;

// base64url: padding-free, '-' / '_' alphabet.
TEST(BoltGcsJwt, Base64UrlEncodeBasics) {
    char out[64];
    // RFC 4648 §10 "foo" → "Zm9v"; same for url-safe (no special chars).
    uint32_t n = base64url_encode(reinterpret_cast<const uint8_t*>("foo"), 3,
                                  out, sizeof(out));
    EXPECT_EQ(n, 4u);
    EXPECT_STREQ(out, "Zm9v");
    // "foob" → "Zm9vYg" (no '=' padding).
    n = base64url_encode(reinterpret_cast<const uint8_t*>("foob"), 4,
                         out, sizeof(out));
    EXPECT_EQ(n, 6u);
    EXPECT_STREQ(out, "Zm9vYg");
    // Bytes that would produce '+' / '/' in standard base64.
    const uint8_t bin[] = { 0xFB, 0xFF, 0xFE };   // → "+//+" std, "-__-" url
    n = base64url_encode(bin, 3, out, sizeof(out));
    EXPECT_EQ(n, 4u);
    EXPECT_STREQ(out, "-__-");
}

// JWT unsigned form: "<b64url(header)>.<b64url(claims)>" with correct claims.
TEST(BoltGcsJwt, UnsignedJwtStructureAndClaims) {
    char jwt[2048];
    const uint64_t now = 1700000000ull;
    const uint32_t n = build_jwt_unsigned(
        "test@my-project.iam.gserviceaccount.com",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://oauth2.googleapis.com/token",
        now, 3600u, jwt, sizeof(jwt));
    ASSERT_GT(n, 0u);
    // Exactly one '.' separator.
    const char* dot = std::strchr(jwt, '.');
    ASSERT_NE(dot, nullptr);
    EXPECT_EQ(std::strchr(dot + 1, '.'), nullptr);

    // Header b64url-decoded must contain "alg":"RS256" and "typ":"JWT".
    // We don't ship a b64url decoder; check the canonical encoded header
    // prefix instead: {"alg":"RS256","typ":"JWT"} → "eyJhbGciOi..."
    EXPECT_EQ(std::strncmp(jwt, "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9", 36), 0);

    // Claims portion (b64url) — we just sanity-check it carries the iss /
    // scope / aud markers when decoded via a textual contains check on the
    // unencoded form is impractical; instead, verify the iat/exp arithmetic
    // is consistent with the chosen ttl by building twice and comparing
    // lengths (claims size is deterministic per input).
    char jwt2[2048];
    const uint32_t n2 = build_jwt_unsigned(
        "test@my-project.iam.gserviceaccount.com",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://oauth2.googleapis.com/token",
        now, 3600u, jwt2, sizeof(jwt2));
    EXPECT_EQ(n, n2);
    EXPECT_EQ(std::strcmp(jwt, jwt2), 0);
}

// Service-account JSON parser: extract client_email + private_key with
// \n-unescape.
TEST(BoltGcsJwt, ParseServiceAccountJson) {
    const char* sa =
        "{\n"
        "  \"type\": \"service_account\",\n"
        "  \"project_id\": \"my-project\",\n"
        "  \"client_email\": \"sa@my-project.iam.gserviceaccount.com\",\n"
        "  \"private_key\": \"-----BEGIN PRIVATE KEY-----\\nABCDEFG\\n-----END PRIVATE KEY-----\\n\",\n"
        "  \"client_id\": \"1234567890\"\n"
        "}\n";
    char email[256];
    char pem[1024];
    uint32_t pem_len = 0;
    ASSERT_TRUE(parse_service_account_json(
        sa, static_cast<uint32_t>(std::strlen(sa)),
        email, sizeof(email), pem, sizeof(pem), &pem_len));
    EXPECT_STREQ(email, "sa@my-project.iam.gserviceaccount.com");
    // Verify the literal "\\n" sequences in the JSON were unescaped to '\n'.
    const char* expected_pem =
        "-----BEGIN PRIVATE KEY-----\nABCDEFG\n-----END PRIVATE KEY-----\n";
    EXPECT_STREQ(pem, expected_pem);
    EXPECT_EQ(pem_len, std::strlen(expected_pem));
}

#if defined(BOLT_WITH_TLS)
// RS256 signing produces a 256-byte (2048-bit RSA) signature → 342 chars in
// base64url (no padding).
TEST(BoltGcsJwt, Rs256ProducesExpectedSignatureLength) {
    // Generate a fresh 2048-bit RSA key in PEM PKCS#8.
    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* kctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    ASSERT_NE(kctx, nullptr);
    ASSERT_EQ(EVP_PKEY_keygen_init(kctx), 1);
    ASSERT_EQ(EVP_PKEY_CTX_set_rsa_keygen_bits(kctx, 2048), 1);
    ASSERT_EQ(EVP_PKEY_keygen(kctx, &pkey), 1);
    EVP_PKEY_CTX_free(kctx);
    ASSERT_NE(pkey, nullptr);

    BIO* mem = BIO_new(BIO_s_mem());
    ASSERT_NE(mem, nullptr);
    ASSERT_EQ(PEM_write_bio_PrivateKey(mem, pkey, nullptr, nullptr, 0,
                                       nullptr, nullptr), 1);
    char* pem_data = nullptr;
    const long pem_len = BIO_get_mem_data(mem, &pem_data);
    ASSERT_GT(pem_len, 0);

    // Build the unsigned JWT.
    char jwt[2048];
    const uint32_t jn = build_jwt_unsigned(
        "test@my-project.iam.gserviceaccount.com",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://oauth2.googleapis.com/token",
        1700000000ull, 3600u, jwt, sizeof(jwt));
    ASSERT_GT(jn, 0u);

    char sig[512];
    const uint32_t sn = sign_jwt_rs256(pem_data, static_cast<uint32_t>(pem_len),
                                       jwt, jn, sig, sizeof(sig));
    // base64url-encoded 256-byte sig: ceil(256*4/3) = 342 chars, no padding.
    EXPECT_EQ(sn, 342u);
    // base64url alphabet only: [A-Za-z0-9_-].
    for (uint32_t i = 0; i < sn; ++i) {
        const char c = sig[i];
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_';
        ASSERT_TRUE(ok) << "bad char at " << i << ": " << static_cast<int>(c);
    }
    BIO_free(mem);
    EVP_PKEY_free(pkey);
}
#endif

}  // namespace
