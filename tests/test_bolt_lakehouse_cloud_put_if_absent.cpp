// Cloud ObjectStores against real emulators: put_if_absent must be an atomic
// create (the Delta/Iceberg commit CAS), plus the get/put/head/list/delete
// transport it rides on. Each suite skips unless its endpoint env var is set:
//   BOLT_TEST_S3_ENDPOINT     e.g. http://localhost:9000      (MinIO, minioadmin)
//   BOLT_TEST_AZURE_ENDPOINT  e.g. http://localhost:10000/devstoreaccount1
//   BOLT_TEST_GCS_ENDPOINT    e.g. http://localhost:4443      (fake-gcs-server)

#include "bolt/crypto/sigv4.h"
#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/azure.h"
#include "bolt/lakehouse/object_store/compat.h"
#include "bolt/lakehouse/object_store/gcs.h"
#include "bolt/net/bolt_http_client.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>

namespace {

using namespace bolt::lakehouse;

constexpr const char* kBucket = "g2ice181";
constexpr const char* kAzAccount = "devstoreaccount1";
constexpr const char* kAzKey =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==";

std::string unique_prefix() {
    const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
    return "t" + std::to_string(static_cast<long long>(ns)) + "/";
}

int send(const char* method, const char* url, const char* body,
         bolt::net::HttpRequest* req) {
    std::strncpy(req->method, method, sizeof(req->method) - 1u);
    std::strncpy(req->url, url, sizeof(req->url) - 1u);
    req->body = reinterpret_cast<const uint8_t*>(body);
    req->body_len = static_cast<uint32_t>(std::strlen(body));
    bolt::Arena arena;
    bolt::net::HttpResponse resp;
    if (bolt::net::http_send(&arena, req, &resp) != 0) return -1;
    return resp.status;
}

void make_s3_bucket(const std::string& ep) {
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    const std::string url = ep + "/" + kBucket;
    char host[256];
    bool https = false;
    uint16_t port = 0;
    char path[2048];
    ASSERT_TRUE(bolt::net::http_url_parse(url.c_str(), &https, host,
                                          sizeof(host), &port, path,
                                          sizeof(path)));
    char hash[65];
    bolt::crypto::sha256_empty_hex(hash);
    const std::time_t t = std::time(nullptr);
    std::tm tmv;
    gmtime_r(&t, &tmv);
    char date[20];
    std::snprintf(date, sizeof(date), "%04d%02d%02dT%02d%02d%02dZ",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    bolt::crypto::SigV4Request s{"PUT", host, path, "", "s3", "us-east-1",
                                 date, hash, "minioadmin", "minioadmin",
                                 nullptr};
    char auth[1024];
    char dout[24];
    bolt::crypto::SigV4Result o{auth, sizeof(auth), dout, sizeof(dout)};
    ASSERT_TRUE(bolt::crypto::sigv4_sign(&s, &o));
    bolt::net::http_request_add_header(&req, "X-Amz-Date", date);
    bolt::net::http_request_add_header(&req, "X-Amz-Content-Sha256", hash);
    bolt::net::http_request_add_header(&req, "Authorization", auth);
    bolt::net::http_request_add_header(&req, "Content-Length", "0");
    const int st = send("PUT", url.c_str(), "", &req);
    ASSERT_TRUE(st == 200 || st == 409) << "create bucket status " << st;
}

void make_az_container(const std::string& ep, const azure_blob::AzureBlobStore* az) {
    using namespace azure_blob;
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    const std::string url = ep + "/" + kBucket + "?restype=container";
    const std::string path = "/" + std::string(kAzAccount) + "/" + kBucket;
    char date[64];
    const std::time_t t = std::time(nullptr);
    std::tm tmv;
    gmtime_r(&t, &tmv);
    std::strftime(date, sizeof(date), "%a, %d %b %Y %H:%M:%S GMT", &tmv);
    SignHeaders h;
    std::memset(&h, 0, sizeof(h));
    CanonicalizedHeader xh[2] = {{"x-ms-date", date},
                                 {"x-ms-version", "2021-08-06"}};
    char sts[2048];
    const uint32_t sl = build_string_to_sign("PUT", &h, kAzAccount,
                                             path.c_str(), xh, 2,
                                             "restype:container", sts,
                                             sizeof(sts));
    ASSERT_GT(sl, 0u);
    char sig[64];
    ASSERT_TRUE(sign_shared_key(az->decoded_key, az->decoded_key_len, sts, sl,
                                sig, sizeof(sig)));
    const std::string auth = std::string("SharedKey ") + kAzAccount + ":" + sig;
    bolt::net::http_request_add_header(&req, "x-ms-date", date);
    bolt::net::http_request_add_header(&req, "x-ms-version", "2021-08-06");
    bolt::net::http_request_add_header(&req, "Authorization", auth.c_str());
    bolt::net::http_request_add_header(&req, "Content-Length", "0");
    const int st = send("PUT", url.c_str(), "", &req);
    ASSERT_TRUE(st == 201 || st == 409) << "create container status " << st;
}

void make_gcs_bucket(const std::string& ep) {
    bolt::net::HttpRequest req;
    std::memset(&req, 0, sizeof(req));
    bolt::net::http_request_add_header(&req, "Content-Type",
                                       "application/json");
    const std::string url = ep + "/storage/v1/b";
    const std::string body = std::string("{\"name\":\"") + kBucket + "\"}";
    const int st = send("POST", url.c_str(), body.c_str(), &req);
    ASSERT_TRUE(st == 200 || st == 409) << "create bucket status " << st;
}

std::string get_str(ObjectStore* os, const std::string& key, int* rc) {
    bolt::Arena arena;
    const uint8_t* d = nullptr;
    uint64_t n = 0;
    *rc = os_get(os, key.c_str(), &arena, &d, &n);
    return *rc == kOsOk ? std::string(reinterpret_cast<const char*>(d), n)
                        : std::string();
}

int put_str(ObjectStore* os, const std::string& key, const std::string& v,
            bool if_absent) {
    const auto* d = reinterpret_cast<const uint8_t*>(v.data());
    return if_absent ? os_put_if_absent(os, key.c_str(), d, v.size())
                     : os_put(os, key.c_str(), d, v.size());
}

// The contract every commit relies on, end to end against a live store.
void exercise(ObjectStore* os) {
    const std::string pre = unique_prefix();
    const std::string key = pre + "_delta_log/00000000000000000001.json";
    int rc = 0;

    ASSERT_EQ(put_str(os, key, "first", true), kOsOk);
    EXPECT_EQ(put_str(os, key, "second", true), kOsExists);
    EXPECT_EQ(get_str(os, key, &rc), "first");
    EXPECT_EQ(rc, kOsOk);

    ObjectMeta m;
    ASSERT_EQ(os_head(os, key.c_str(), &m), kOsOk);
    EXPECT_TRUE(m.exists);
    EXPECT_EQ(m.size, 5u);
    EXPECT_EQ(os_head(os, (pre + "missing").c_str(), &m), kOsNotFound);
    (void)get_str(os, pre + "missing", &rc);
    EXPECT_EQ(rc, kOsNotFound);

    ASSERT_EQ(put_str(os, pre + "a b+c.json", "x", false), kOsOk);
    static ObjectEntry entries[16];
    uint32_t n = 0;
    ASSERT_EQ(os_list(os, pre.c_str(), entries, 16, &n), kOsOk);
    ASSERT_EQ(n, 2u);
    bool saw_key = false;
    bool saw_odd = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (key == entries[i].key) { saw_key = true; EXPECT_EQ(entries[i].size, 5u); }
        if (pre + "a b+c.json" == entries[i].key) saw_odd = true;
    }
    EXPECT_TRUE(saw_key);
    EXPECT_TRUE(saw_odd);

    ASSERT_EQ(put_str(os, key, "overwritten", false), kOsOk);
    EXPECT_EQ(get_str(os, key, &rc), "overwritten");
    ASSERT_EQ(os_delete(os, key.c_str()), kOsOk);
    EXPECT_EQ(os_head(os, key.c_str(), &m), kOsNotFound);
    EXPECT_EQ(put_str(os, key, "", true), kOsOk);
    EXPECT_EQ(put_str(os, key, "", true), kOsExists);

    // Many writers race for one commit slot: exactly one may win, and the
    // stored body must be the winner's.
    const std::string slot = pre + "metadata/v2.metadata.json";
    constexpr int kWriters = 8;
    std::atomic<int> wins{0};
    std::atomic<int> exists{0};
    std::atomic<int> winner{-1};
    std::thread th[kWriters];
    for (int w = 0; w < kWriters; ++w) {
        th[w] = std::thread([&, w] {
            const int r = put_str(os, slot, "writer-" + std::to_string(w), true);
            if (r == kOsOk) { wins.fetch_add(1); winner.store(w); }
            else if (r == kOsExists) exists.fetch_add(1);
        });
    }
    for (auto& t : th) t.join();
    EXPECT_EQ(wins.load(), 1);
    EXPECT_EQ(exists.load(), kWriters - 1);
    ASSERT_GE(winner.load(), 0);
    EXPECT_EQ(get_str(os, slot, &rc), "writer-" + std::to_string(winner.load()));
}

TEST(BoltCloudPutIfAbsent, S3MinIO) {
    const char* ep = std::getenv("BOLT_TEST_S3_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_S3_ENDPOINT unset";
    make_s3_bucket(ep);
    bolt::Arena arena;
    s3_compat::Secret k;
    s3_compat::Secret s;
    ASSERT_TRUE(s3_compat::secret_set(&k, "minioadmin"));
    ASSERT_TRUE(s3_compat::secret_set(&s, "minioadmin"));
    ObjectStore os;
    ASSERT_TRUE(s3_compat::minio_store_new(&os, &arena, ep, kBucket, &k, &s));
    exercise(&os);
}

TEST(BoltCloudPutIfAbsent, S3WrongSecretIsRefused) {
    const char* ep = std::getenv("BOLT_TEST_S3_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_S3_ENDPOINT unset";
    make_s3_bucket(ep);
    bolt::Arena arena;
    s3_compat::Secret k;
    s3_compat::Secret s;
    ASSERT_TRUE(s3_compat::secret_set(&k, "minioadmin"));
    ASSERT_TRUE(s3_compat::secret_set(&s, "not-the-secret"));
    ObjectStore os;
    ASSERT_TRUE(s3_compat::minio_store_new(&os, &arena, ep, kBucket, &k, &s));
    const std::string key = unique_prefix() + "x";
    EXPECT_EQ(put_str(&os, key, "v", true), kOsIoError);
}

TEST(BoltCloudPutIfAbsent, AzureAzurite) {
    const char* ep = std::getenv("BOLT_TEST_AZURE_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_AZURE_ENDPOINT unset";
    bolt::Arena arena;
    azure_blob::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.account, kAzAccount, sizeof(cfg.account) - 1u);
    ASSERT_TRUE(s3_compat::secret_set(&cfg.account_key, kAzKey));
    std::strncpy(cfg.container, kBucket, sizeof(cfg.container) - 1u);
    std::strncpy(cfg.endpoint_override, ep, sizeof(cfg.endpoint_override) - 1u);
    ObjectStore os;
    ASSERT_TRUE(azure_blob::azure_blob_store_new(&os, &arena, &cfg));
    make_az_container(ep, static_cast<const azure_blob::AzureBlobStore*>(os.impl));
    exercise(&os);
}

TEST(BoltCloudPutIfAbsent, GcsFakeServer) {
    const char* ep = std::getenv("BOLT_TEST_GCS_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_GCS_ENDPOINT unset";
    make_gcs_bucket(ep);
    bolt::Arena arena;
    static gcs::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.bucket, kBucket, sizeof(cfg.bucket) - 1u);
    cfg.auth_mode = gcs::AuthMode::Hmac;
    std::strncpy(cfg.endpoint_override, ep, sizeof(cfg.endpoint_override) - 1u);
    ObjectStore os;
    ASSERT_TRUE(gcs::gcs_store_new(&os, &arena, &cfg));
    exercise(&os);
}

// An HMAC access id without its secret is a config error, never anonymous.
TEST(BoltCloudPutIfAbsent, GcsHmacWithoutSecretIsRejected) {
    bolt::Arena arena;
    static gcs::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.bucket, kBucket, sizeof(cfg.bucket) - 1u);
    cfg.auth_mode = gcs::AuthMode::Hmac;
    ASSERT_TRUE(s3_compat::secret_set(&cfg.hmac_access_id, "GOOG1EXAMPLE"));
    ObjectStore os;
    EXPECT_FALSE(gcs::gcs_store_new(&os, &arena, &cfg));
}

// GCS HMAC keys sign XML API requests with AWS4-HMAC-SHA256 (region "auto").
// fake-gcs-server has no XML PUT and checks no signatures, so the signing is
// proven against MinIO, which verifies every one. MinIO ignores
// x-goog-if-generation-match, so the CAS itself is not exercised here.
gcs::Config* gcs_hmac_cfg(const char* ep, const char* secret) {
    static gcs::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.bucket, kBucket, sizeof(cfg.bucket) - 1u);
    cfg.auth_mode = gcs::AuthMode::Hmac;
    std::strncpy(cfg.endpoint_override, ep, sizeof(cfg.endpoint_override) - 1u);
    if (!s3_compat::secret_set(&cfg.hmac_access_id, "minioadmin") ||
        !s3_compat::secret_set(&cfg.hmac_secret, secret)) return nullptr;
    return &cfg;
}

TEST(BoltCloudPutIfAbsent, GcsHmacSigningAgainstMinIO) {
    const char* ep = std::getenv("BOLT_TEST_S3_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_S3_ENDPOINT unset";
    make_s3_bucket(ep);
    bolt::Arena arena;
    gcs::Config* cfg = gcs_hmac_cfg(ep, "minioadmin");
    ASSERT_NE(cfg, nullptr);
    ObjectStore os;
    ASSERT_TRUE(gcs::gcs_store_new(&os, &arena, cfg));
    const std::string pre = unique_prefix();
    const std::string key = pre + "_delta_log/00000000000000000001.json";
    int rc = 0;
    ASSERT_EQ(put_str(&os, key, "first", true), kOsOk);
    EXPECT_EQ(get_str(&os, key, &rc), "first");
    EXPECT_EQ(rc, kOsOk);
    ObjectMeta m;
    ASSERT_EQ(os_head(&os, key.c_str(), &m), kOsOk);
    EXPECT_EQ(m.size, 5u);
    ASSERT_EQ(put_str(&os, pre + "a b+c.json", "x", false), kOsOk);
    static ObjectEntry entries[16];
    uint32_t n = 0;
    ASSERT_EQ(os_list(&os, pre.c_str(), entries, 16, &n), kOsOk);
    EXPECT_EQ(n, 2u);
    ASSERT_EQ(os_delete(&os, key.c_str()), kOsOk);
    EXPECT_EQ(os_head(&os, key.c_str(), &m), kOsNotFound);
}

TEST(BoltCloudPutIfAbsent, GcsHmacWrongSecretIsRefused) {
    const char* ep = std::getenv("BOLT_TEST_S3_ENDPOINT");
    if (ep == nullptr || ep[0] == '\0') GTEST_SKIP() << "BOLT_TEST_S3_ENDPOINT unset";
    make_s3_bucket(ep);
    bolt::Arena arena;
    gcs::Config* cfg = gcs_hmac_cfg(ep, "not-the-secret");
    ASSERT_NE(cfg, nullptr);
    ObjectStore os;
    ASSERT_TRUE(gcs::gcs_store_new(&os, &arena, cfg));
    EXPECT_EQ(put_str(&os, unique_prefix() + "x", "v", true), kOsIoError);
}

}  // namespace
