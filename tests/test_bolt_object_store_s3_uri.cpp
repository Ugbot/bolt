#include "bolt/lakehouse/object_store/s3_uri.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <string>

namespace {
using namespace bolt::lakehouse;
using Status = S3UriStatus;

TEST(S3ExactUri, CanonicalRoundTripDecodesExactlyOnce) {
    struct Case { const char* root; const char* key; const char* uri; };
    const Case cases[]{
        {"s3://bucket", "outside/table.avro", "s3://bucket/outside/table.avro"},
        {"s3://bucket/root", "a b/plus+?.json", "s3://bucket/root/a%20b/plus%2B%3F.json"},
        {"s3://bucket/root%20space", "percent%2F", "s3://bucket/root%20space/percent%252F"},
        {"s3://bucket", "utf8/\xC3\xA9", "s3://bucket/utf8/%C3%A9"},
        {"s3://bucket", "hash#colon:", "s3://bucket/hash%23colon%3A"}
    };
    for (const auto& item : cases) {
        std::array<char, kOsMaxKey> out{};
        ASSERT_EQ(s3_uri_for_object_key(item.root, item.key, out.data(), out.size()), Status::Ok);
        EXPECT_STREQ(out.data(), item.uri);
        ASSERT_EQ(s3_uri_object_key(item.root, item.uri, out.data(), out.size()), Status::Ok);
        EXPECT_STREQ(out.data(), item.key);
    }
}

TEST(S3ExactUri, RefusesAuthorityPrefixAndTableRebaseAmbiguity) {
    for (const char* uri : {"s3://other/root/data/a", "s3://bucket/rooted/a",
                            "s3://bucket/root", "s3://bucket/a"}) {
        std::array<char, kOsMaxKey> out{}; out.fill('~'); const auto before = out;
        EXPECT_EQ(s3_uri_object_key("s3://bucket/root", uri, out.data(), out.size()), Status::OutsideRoot);
        EXPECT_EQ(out, before);
    }
    std::array<char, kOsMaxKey> key{};
    EXPECT_EQ(s3_uri_object_key("s3://bucket", "s3://bucket/outside/table/file.parquet",
                               key.data(), key.size()), Status::Ok);
    EXPECT_STREQ(key.data(), "outside/table/file.parquet");
}

TEST(S3ExactUri, UnsupportedSchemesNeverBecomeStoreKeys) {
    for (const char* uri : {"https://bucket/root/a", "gs://bucket/root/a",
                            "file:///root/a", "root/a", ""}) {
        char out[32] = "unchanged";
        EXPECT_EQ(s3_uri_validate(uri), Status::Unsupported);
        EXPECT_EQ(s3_uri_object_key("s3://bucket/root", uri, out, sizeof(out)), Status::Unsupported);
        EXPECT_STREQ(out, "unchanged");
    }
}

// G2ICE-300: a recorded s3a:// or s3n:// path names the same object as its s3:// spelling; a configured root may not
// use the alias, and an alias outside the root is still outside it.
TEST(S3ExactUri, HadoopSchemeAliasesMapExactly) {
    for (const char* uri : {"s3a://bucket/root/data/a.parquet", "s3n://bucket/root/data/a.parquet"}) {
        std::array<char, kOsMaxKey> key{};
        EXPECT_EQ(s3_uri_object_key("s3://bucket/root", uri, key.data(), key.size()), Status::Ok) << uri;
        EXPECT_STREQ(key.data(), "data/a.parquet");
        EXPECT_EQ(s3_uri_validate(uri), Status::Unsupported) << "validate stays canonical: " << uri;
    }
    std::array<char, kOsMaxKey> part{};                                  // Spark's partition directories: '=' stays literal
    EXPECT_EQ(s3_uri_object_key("s3://bucket/root", "s3a://bucket/root/data/p=1/b=2,c/f.parquet", part.data(), part.size()), Status::Ok);
    EXPECT_STREQ(part.data(), "data/p=1/b=2,c/f.parquet");
    char out[32] = "unchanged";
    EXPECT_EQ(s3_uri_object_key("s3a://bucket/root", "s3://bucket/root/a", out, sizeof(out)), Status::Unsupported);
    EXPECT_EQ(s3_uri_object_key("s3://bucket/root", "s3a://other/root/a", out, sizeof(out)), Status::OutsideRoot);
    EXPECT_EQ(s3_uri_object_key("s3://bucket/root", "s3a://bucket/root/a%2fb", out, sizeof(out)), Status::BadArg);
    EXPECT_STREQ(out, "unchanged");
}

TEST(S3ExactUri, NoncanonicalAndTraversalFormsRefuseWithoutOutput) {
    for (const char* uri : {"s3://Bucket/a", "s3://ab/a", "s3://-bucket/a", "s3://bucket./a",
             "s3://bucket/", "s3://bucket//a", "s3://bucket/a/../b", "s3://bucket/./b",
             "s3://bucket/a?x=1", "s3://bucket/a#x", "s3://bucket/a%2fb", "s3://bucket/a%2Fb",
             "s3://bucket/%61", "s3://bucket/%2E%2E/b", "s3://bucket/a%5Cb", "s3://bucket/%00",
             "s3://bucket/%7F", "s3://bucket/%", "s3://bucket/%0G", "s3://bucket/a b"}) {
        char out[64] = "unchanged";
        EXPECT_EQ(s3_uri_validate(uri), Status::BadArg) << uri;
        EXPECT_EQ(s3_uri_object_key("s3://bucket", uri, out, sizeof(out)), Status::BadArg) << uri;
        EXPECT_STREQ(out, "unchanged");
    }
}

TEST(S3ExactUri, InvalidRawKeysNeverEncodeASecondAuthority) {
    for (const char* key : {"", "/absolute", "a//b", "a/../b", "./a", "a/", "a\\b", "a\nb"}) {
        char out[64] = "unchanged";
        EXPECT_EQ(s3_uri_for_object_key("s3://bucket", key, out, sizeof(out)), Status::BadArg);
        EXPECT_STREQ(out, "unchanged");
    }
}

TEST(S3ExactUri, ExactCapacityAndUnterminatedBounds) {
    char tiny[4] = "old";
    EXPECT_EQ(s3_uri_object_key("s3://bucket", "s3://bucket/abcd", tiny, sizeof(tiny)), Status::Capacity);
    EXPECT_STREQ(tiny, "old");
    char exact[5]{};
    ASSERT_EQ(s3_uri_object_key("s3://bucket", "s3://bucket/abcd", exact, sizeof(exact)), Status::Ok);
    EXPECT_STREQ(exact, "abcd");
    EXPECT_EQ(s3_uri_for_object_key("s3://bucket", "a b", tiny, sizeof(tiny)), Status::Capacity);
    EXPECT_STREQ(tiny, "old");
    std::array<char, kOsMaxKey> unterminated{}; unterminated.fill('a');
    EXPECT_EQ(s3_uri_validate(unterminated.data()), Status::BadArg);
    EXPECT_EQ(s3_uri_object_key("s3://bucket", unterminated.data(), exact, sizeof(exact)), Status::BadArg);
    EXPECT_STREQ(exact, "abcd");
    EXPECT_EQ(s3_uri_validate(nullptr), Status::BadArg);
    EXPECT_EQ(s3_uri_for_object_key(nullptr, "a", exact, sizeof(exact)), Status::BadArg);
}

TEST(S3ExactUri, AliasRefusalHappensBeforeAnyWrite) {
    std::array<char, 128> root{}; std::strcpy(root.data(), "s3://bucket/root");
    const auto before = root;
    EXPECT_EQ(s3_uri_for_object_key(root.data(), "file", root.data(), root.size()), Status::BadArg);
    EXPECT_EQ(root, before);
    std::array<char, 128> uri{}; std::strcpy(uri.data(), "s3://bucket/root/file");
    const auto old_uri = uri;
    EXPECT_EQ(s3_uri_object_key(root.data(), uri.data(), uri.data(), uri.size()), Status::BadArg);
    EXPECT_EQ(uri, old_uri);
    EXPECT_EQ(s3_uri_for_object_key(root.data(), "file", nullptr, 1), Status::BadArg);
    char out[4] = "old";
    EXPECT_EQ(s3_uri_for_object_key(root.data(), "file", out, kOsMaxKey + 1), Status::BadArg);
    EXPECT_STREQ(out, "old");
}
} // namespace
