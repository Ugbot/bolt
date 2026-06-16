// Azure Blob Shared Key signing — base64 + HMAC + canonical StringToSign.
//
// References:
//   https://learn.microsoft.com/rest/api/storageservices/authorize-with-shared-key
// We don't have a single published end-to-end MS test vector with all fields
// pinned, so we verify the building blocks independently:
//   1. Base64 round-trip (RFC 4648 §10 standard vectors).
//   2. Base64 decode of a real-shape Azure portal key works.
//   3. The canonical StringToSign matches the MS canonical example for a
//      GET https://myaccount.blob.core.windows.net/mycontainer?comp=metadata.
//   4. HMAC-SHA256 + base64 produces the expected signature against a known
//      key/StringToSign pair (RFC-style: known key, known message → known mac).

#include "bolt/lakehouse/object_store/azure.h"
#include "bolt/lakehouse/object_store/compat.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace {

using namespace bolt::lakehouse::azure_blob;

// RFC 4648 §10 base64 test vectors.
TEST(BoltAzureSign, Base64RoundTripRfc4648) {
    struct V { const char* plain; const char* b64; };
    const V cases[] = {
        {"",       ""},
        {"f",      "Zg=="},
        {"fo",     "Zm8="},
        {"foo",    "Zm9v"},
        {"foob",   "Zm9vYg=="},
        {"fooba",  "Zm9vYmE="},
        {"foobar", "Zm9vYmFy"},
    };
    for (const auto& c : cases) {
        char enc[32];
        const uint32_t en = base64_encode(
            reinterpret_cast<const uint8_t*>(c.plain),
            static_cast<uint32_t>(std::strlen(c.plain)), enc, sizeof(enc));
        ASSERT_EQ(en, std::strlen(c.b64)) << "encode " << c.plain;
        EXPECT_STREQ(enc, c.b64);
        uint8_t dec[16] = {};
        const uint32_t dn = base64_decode(c.b64, dec, sizeof(dec));
        ASSERT_EQ(dn, std::strlen(c.plain)) << "decode " << c.b64;
        EXPECT_EQ(std::memcmp(dec, c.plain, dn), 0);
    }
}

// HMAC-SHA256 over a known StringToSign with a known decoded key produces a
// known base64 signature. We use a key the user can verify with:
//   echo -n "<sts>" | openssl dgst -sha256 -binary \
//                     -hmac "$(printf 'YWJj' | base64 -d)" | base64
// i.e. raw key bytes = "abc".
TEST(BoltAzureSign, SharedKeyHmacKnownVector) {
    const uint8_t key[] = { 'a', 'b', 'c' };
    const char sts[] = "GET\n\n\n\n\n\n\n\n\n\n\n\n/myaccount/mycontainer\ncomp:list";
    char sig[64];
    ASSERT_TRUE(sign_shared_key(key, sizeof(key), sts,
                                static_cast<uint32_t>(std::strlen(sts)),
                                sig, sizeof(sig)));
    // Computed independently via openssl: HMAC-SHA256(key="abc", msg=sts)
    // → base64. (See file-level comment for the reproduction recipe.)
    // The expected value below was computed with python's hmac+base64.
    // import hmac, hashlib, base64
    // base64.b64encode(hmac.new(b'abc', sts.encode(), hashlib.sha256).digest())
    EXPECT_STREQ(sig, "NQz3yBjPUiUzQYg8av1iR0Tjmsn897npEYdB86SvaHU=");
}

// Canonical StringToSign matches the MS canonical example. We use the
// "List Containers" shape:
//   GET /\?comp=list with x-ms-date + x-ms-version headers, no body.
TEST(BoltAzureSign, CanonicalStringToSignListContainers) {
    SignHeaders h{};
    h.content_encoding    = "";
    h.content_language    = "";
    h.content_length      = "";        // empty for zero-byte body
    h.content_md5         = "";
    h.content_type        = "";
    h.date_header         = "";        // we use x-ms-date instead
    h.if_modified_since   = "";
    h.if_match            = "";
    h.if_none_match       = "";
    h.if_unmodified_since = "";
    h.range               = "";

    const CanonicalizedHeader xms[] = {
        {"x-ms-date",    "Fri, 26 Jun 2015 23:39:12 GMT"},
        {"x-ms-version", "2015-02-21"},
    };
    char sts[2048];
    const uint32_t n = build_string_to_sign(
        "GET", &h, "myaccount", "/", xms, 2u, "comp:list",
        sts, sizeof(sts));
    ASSERT_GT(n, 0u);

    // Per MS docs the StringToSign must be exactly this:
    const char* expected =
        "GET\n\n\n\n\n\n\n\n\n\n\n\n"
        "x-ms-date:Fri, 26 Jun 2015 23:39:12 GMT\n"
        "x-ms-version:2015-02-21\n"
        "/myaccount/\ncomp:list";
    EXPECT_STREQ(sts, expected);
    EXPECT_EQ(n, std::strlen(expected));
}

// azure_blob_store_new round-trips Config + decodes the base64 account key.
TEST(BoltAzureSign, StoreInitDecodesPortalKey) {
    Config cfg{};
    std::strncpy(cfg.account, "myacct", sizeof(cfg.account) - 1);
    // Portal key = base64 of "supersecretkey" (14 bytes).
    bolt::lakehouse::s3_compat::secret_set(&cfg.account_key, "c3VwZXJzZWNyZXRrZXk=");
    std::strncpy(cfg.container, "mycontainer", sizeof(cfg.container) - 1);
    cfg.endpoint_override[0] = '\0';
    cfg.sas_token[0] = '\0';

    bolt::Arena a;
    bolt::lakehouse::ObjectStore os{};
    ASSERT_TRUE(azure_blob_store_new(&os, &a, &cfg));
    ASSERT_NE(os.impl, nullptr);
    const auto* store = static_cast<const AzureBlobStore*>(os.impl);
    EXPECT_EQ(store->decoded_key_len, 14u);
    EXPECT_EQ(std::memcmp(store->decoded_key, "supersecretkey", 14u), 0);
    EXPECT_FALSE(store->use_sas);
}

}  // namespace
