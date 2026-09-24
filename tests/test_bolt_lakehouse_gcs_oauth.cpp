// GCS service-account auth end to end against an in-process OAuth token
// endpoint + JSON-API fake. The fake verifies each JWT's RS256 signature with
// the service account's public key and its claims, issues bearer tokens, and
// refuses storage requests that do not carry an issued token — so a pass means
// the store minted, cached, refreshed and sent real credentials.

#include "bolt/lakehouse/object_store.h"
#include "bolt/lakehouse/object_store/compat.h"
#include "bolt/lakehouse/object_store/gcs.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#if defined(BOLT_WITH_TLS) && !defined(_WIN32)
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  include <openssl/bio.h>
#  include <openssl/evp.h>
#  include <openssl/pem.h>
#  include <openssl/rsa.h>
#  define G2ICE190_FAKE 1
#endif

#if defined(G2ICE190_FAKE)
namespace {

using namespace bolt::lakehouse;

constexpr const char* kEmail = "sa@g2ice190.iam.gserviceaccount.com";
constexpr const char* kScope =
    "https://www.googleapis.com/auth/devstorage.read_write";

EVP_PKEY* new_rsa() {
    EVP_PKEY* k = nullptr;
    EVP_PKEY_CTX* c = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (c == nullptr) return nullptr;
    if (EVP_PKEY_keygen_init(c) != 1 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(c, 2048) != 1 ||
        EVP_PKEY_keygen(c, &k) != 1) k = nullptr;
    EVP_PKEY_CTX_free(c);
    return k;
}

std::string pem_of(EVP_PKEY* k) {
    BIO* m = BIO_new(BIO_s_mem());
    PEM_write_bio_PrivateKey(m, k, nullptr, nullptr, 0, nullptr, nullptr);
    char* d = nullptr;
    const long n = BIO_get_mem_data(m, &d);
    std::string s(d, static_cast<size_t>(n));
    BIO_free(m);
    return s;
}

std::string b64url_decode(const std::string& in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '-') return 62;
        if (c == '_') return 63;
        return -1;
    };
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        const int v = val(c);
        if (v < 0) return std::string();
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

std::string pct_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            o.push_back(static_cast<char>(
                std::stoi(s.substr(i + 1, 2), nullptr, 16)));
            i += 2;
        } else {
            o.push_back(s[i]);
        }
    }
    return o;
}

std::string query_param(const std::string& target, const std::string& name) {
    const size_t q = target.find('?');
    if (q == std::string::npos) return std::string();
    const std::string key = name + "=";
    size_t p = q + 1;
    while (p < target.size()) {
        size_t e = target.find('&', p);
        if (e == std::string::npos) e = target.size();
        if (target.compare(p, key.size(), key) == 0) {
            return pct_decode(target.substr(p + key.size(),
                                            e - p - key.size()));
        }
        p = e + 1;
    }
    return std::string();
}

uint64_t claim_uint(const std::string& json, const std::string& field) {
    const size_t p = json.find("\"" + field + "\":");
    if (p == std::string::npos) return 0;
    return std::strtoull(json.c_str() + p + field.size() + 3, nullptr, 10);
}

class FakeGoogle {
public:
    EVP_PKEY* verify_key = nullptr;
    std::string token_uri;
    uint64_t expires_in = 3600;
    bool refuse = false;
    std::string token_override;
    std::atomic<int> mints{0};
    std::atomic<int> storage_requests{0};
    std::atomic<int> storage_unauth{0};
    std::string last_jwt_error;

    bool start() {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
            ::listen(fd_, 64) != 0) return false;
        socklen_t al = sizeof(a);
        ::getsockname(fd_, reinterpret_cast<sockaddr*>(&a), &al);
        port_ = ntohs(a.sin_port);
        token_uri = "http://127.0.0.1:" + std::to_string(port_) + "/token";
        th_ = std::thread([this] { serve(); });
        return true;
    }
    ~FakeGoogle() {
        stop_.store(true);
        if (th_.joinable()) th_.join();
        if (fd_ >= 0) ::close(fd_);
    }
    std::string endpoint() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

private:
    int fd_ = -1;
    uint16_t port_ = 0;
    std::thread th_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::set<std::string> issued_;
    std::map<std::string, std::string> objects_;

    void serve() {
        while (!stop_.load()) {
            pollfd p{fd_, POLLIN, 0};
            if (::poll(&p, 1, 50) <= 0) continue;
            const int c = ::accept(fd_, nullptr, nullptr);
            if (c < 0) continue;
            handle(c);
            ::close(c);
        }
    }

    static bool read_request(int c, std::string* head, std::string* body) {
        std::string buf;
        char tmp[4096];
        size_t he = std::string::npos;
        while (he == std::string::npos && buf.size() < (1u << 20)) {
            const ssize_t n = ::recv(c, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, static_cast<size_t>(n));
            he = buf.find("\r\n\r\n");
        }
        if (he == std::string::npos) return false;
        *head = buf.substr(0, he);
        size_t cl = 0;
        const size_t p = head->find("Content-Length:");
        if (p != std::string::npos) {
            cl = std::strtoull(head->c_str() + p + 15, nullptr, 10);
        }
        *body = buf.substr(he + 4);
        while (body->size() < cl) {
            const ssize_t n = ::recv(c, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            body->append(tmp, static_cast<size_t>(n));
        }
        return true;
    }

    static void reply(int c, int status, const std::string& body) {
        char h[256];
        const int n = std::snprintf(h, sizeof(h),
            "HTTP/1.1 %d X\r\nContent-Type: application/json\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n",
            status, body.size());
        std::string out(h, static_cast<size_t>(n));
        out += body;
        size_t off = 0;
        while (off < out.size()) {
            const ssize_t w = ::send(c, out.data() + off, out.size() - off, 0);
            if (w <= 0) return;
            off += static_cast<size_t>(w);
        }
    }

    void handle(int c) {
        std::string head;
        std::string body;
        if (!read_request(c, &head, &body)) return;
        const size_t sp1 = head.find(' ');
        const size_t sp2 = head.find(' ', sp1 + 1);
        const std::string method = head.substr(0, sp1);
        const std::string target = head.substr(sp1 + 1, sp2 - sp1 - 1);
        if (target.rfind("/token", 0) == 0) return token(c, method, body);
        if (head.find("AWS4-HMAC-SHA256") != std::string::npos) {
            return xml(c, head, method, target, body);
        }
        storage(c, head, method, target, body);
    }

    // Returns "" when the assertion is valid, else why not.
    std::string check_jwt(const std::string& jwt) {
        const size_t d1 = jwt.find('.');
        const size_t d2 = jwt.find('.', d1 + 1);
        if (d1 == std::string::npos || d2 == std::string::npos) return "shape";
        const std::string header = b64url_decode(jwt.substr(0, d1));
        const std::string claims = b64url_decode(jwt.substr(d1 + 1, d2 - d1 - 1));
        const std::string sig = b64url_decode(jwt.substr(d2 + 1));
        if (header != "{\"alg\":\"RS256\",\"typ\":\"JWT\"}") return "header";
        if (claims.find(std::string("\"iss\":\"") + kEmail + "\"") ==
            std::string::npos) return "iss";
        if (claims.find("\"aud\":\"" + token_uri + "\"") == std::string::npos)
            return "aud";
        if (claims.find(std::string("\"scope\":\"") + kScope + "\"") ==
            std::string::npos) return "scope";
        const uint64_t iat = claim_uint(claims, "iat");
        const uint64_t exp = claim_uint(claims, "exp");
        const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
        if (exp != iat + 3600 || iat + 300 < now || iat > now + 300)
            return "iat/exp";
        EVP_MD_CTX* m = EVP_MD_CTX_new();
        int ok = EVP_DigestVerifyInit(m, nullptr, EVP_sha256(), nullptr,
                                      verify_key);
        if (ok == 1) {
            ok = EVP_DigestVerify(
                m, reinterpret_cast<const unsigned char*>(sig.data()),
                sig.size(), reinterpret_cast<const unsigned char*>(jwt.data()),
                d2);
        }
        EVP_MD_CTX_free(m);
        return ok == 1 ? std::string() : std::string("signature");
    }

    void token(int c, const std::string& method, const std::string& body) {
        const std::string grant =
            "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
            "&assertion=";
        std::string why;
        if (method != "POST" || body.rfind(grant, 0) != 0) why = "grant";
        else why = check_jwt(body.substr(grant.size()));
        if (why.empty() && refuse) why = "refused";
        if (!why.empty()) {
            {
                std::lock_guard<std::mutex> g(mu_);
                last_jwt_error = why;
            }
            return reply(c, 400, "{\"error\":\"invalid_grant\","
                                 "\"error_description\":\"" + why + "\"}");
        }
        const int n = mints.fetch_add(1) + 1;
        const std::string tok = token_override.empty()
            ? "ya29.fake-" + std::to_string(n) : token_override;
        {
            std::lock_guard<std::mutex> g(mu_);
            issued_.insert(tok);
        }
        reply(c, 200, "{\"access_token\":\"" + tok + "\",\"expires_in\":" +
                          std::to_string(expires_in) +
                          ",\"token_type\":\"Bearer\"}");
    }

    void storage(int c, const std::string& head, const std::string& method,
                 const std::string& target, const std::string& body) {
        storage_requests.fetch_add(1);
        const size_t a = head.find("\r\nAuthorization: Bearer ");
        std::string tok;
        if (a != std::string::npos) {
            const size_t s = a + 24;
            tok = head.substr(s, head.find("\r\n", s) - s);
        }
        std::lock_guard<std::mutex> g(mu_);
        if (issued_.count(tok) == 0) {
            storage_unauth.fetch_add(1);
            return reply(c, 401, "{\"error\":\"unauthenticated\"}");
        }
        const std::string pfx = "/storage/v1/b/bkt/o/";
        if (method == "POST" && target.rfind("/upload/storage/v1/b/bkt/o?", 0) == 0) {
            const std::string name = query_param(target, "name");
            if (query_param(target, "ifGenerationMatch") == "0" &&
                objects_.count(name) != 0) return reply(c, 412, "{}");
            objects_[name] = body;
            return reply(c, 200, "{\"name\":\"" + name + "\"}");
        }
        if (method == "GET" && target.rfind(pfx, 0) == 0) {
            const size_t q = target.find('?');
            const std::string name =
                pct_decode(target.substr(pfx.size(), q - pfx.size()));
            auto it = objects_.find(name);
            if (it == objects_.end()) return reply(c, 404, "{}");
            return reply(c, 200, it->second);
        }
        reply(c, 400, "{}");
    }

public:
    std::atomic<int> xml_cas_puts{0};
    std::atomic<int> xml_unsigned_cas{0};

private:
    // XML API (HMAC): honours x-goog-if-generation-match: 0, and counts
    // CAS PUTs whose precondition header was not covered by the signature.
    void xml(int c, const std::string& head, const std::string& method,
             const std::string& target, const std::string& body) {
        std::lock_guard<std::mutex> g(mu_);
        const std::string pfx = "/bkt/";
        if (target.rfind(pfx, 0) != 0) return reply(c, 400, "{}");
        const std::string name = pct_decode(target.substr(pfx.size()));
        if (method == "PUT") {
            const bool cas = head.find("\r\nx-goog-if-generation-match: 0") !=
                             std::string::npos;
            if (cas) {
                xml_cas_puts.fetch_add(1);
                if (head.find("x-goog-if-generation-match,") ==
                    std::string::npos) xml_unsigned_cas.fetch_add(1);
                if (objects_.count(name) != 0) return reply(c, 412, "{}");
            }
            objects_[name] = body;
            return reply(c, 200, "");
        }
        if (method == "GET") {
            auto it = objects_.find(name);
            if (it == objects_.end()) return reply(c, 404, "");
            return reply(c, 200, it->second);
        }
        reply(c, 400, "{}");
    }
};

std::string sa_json(const std::string& pem, const std::string& token_uri) {
    std::string esc;
    for (char ch : pem) {
        if (ch == '\n') esc += "\\n";
        else esc.push_back(ch);
    }
    std::string j = std::string("{\"type\":\"service_account\",") +
                    "\"client_email\":\"" + kEmail + "\"," +
                    "\"private_key\":\"" + esc + "\"";
    if (!token_uri.empty()) j += ",\"token_uri\":\"" + token_uri + "\"";
    return j + "}";
}

struct Fixture {
    EVP_PKEY* key = new_rsa();
    FakeGoogle fake;
    bolt::Arena arena;
    gcs::Config cfg;
    ObjectStore os{};

    Fixture() { std::memset(&cfg, 0, sizeof(cfg)); }
    ~Fixture() { EVP_PKEY_free(key); }

    bool open(const std::string& token_uri_override = "") {
        if (key == nullptr) return false;
        if (fake.verify_key == nullptr) fake.verify_key = key;
        if (!fake.start()) return false;
        const std::string j = sa_json(pem_of(key),
            token_uri_override.empty() ? fake.token_uri : token_uri_override);
        if (j.size() >= sizeof(cfg.service_account_json.bytes)) return false;
        std::memcpy(cfg.service_account_json.bytes, j.data(), j.size());
        cfg.service_account_json.len = static_cast<uint32_t>(j.size());
        std::strncpy(cfg.bucket, "bkt", sizeof(cfg.bucket) - 1u);
        cfg.auth_mode = gcs::AuthMode::ServiceAccount;
        const std::string ep = fake.endpoint();
        std::strncpy(cfg.endpoint_override, ep.c_str(),
                     sizeof(cfg.endpoint_override) - 1u);
        return gcs::gcs_store_new(&os, &arena, &cfg);
    }
    gcs::GcsStore* store() { return static_cast<gcs::GcsStore*>(os.impl); }
};

int put(ObjectStore* os, const std::string& k, const std::string& v,
        bool if_absent) {
    const auto* d = reinterpret_cast<const uint8_t*>(v.data());
    return if_absent ? os_put_if_absent(os, k.c_str(), d, v.size())
                     : os_put(os, k.c_str(), d, v.size());
}

std::string get(ObjectStore* os, const std::string& k, int* rc) {
    bolt::Arena a;
    const uint8_t* d = nullptr;
    uint64_t n = 0;
    *rc = os_get(os, k.c_str(), &a, &d, &n);
    return *rc == kOsOk ? std::string(reinterpret_cast<const char*>(d), n)
                        : std::string();
}

TEST(BoltGcsOAuth, MintsOnceAndReusesCachedToken) {
    Fixture f;
    ASSERT_TRUE(f.open());
    ASSERT_EQ(put(&f.os, "d/_delta_log/0.json", "first", true), kOsOk)
        << f.store()->last_auth_error << " jwt:" << f.fake.last_jwt_error;
    EXPECT_EQ(put(&f.os, "d/_delta_log/0.json", "second", true), kOsExists);
    int rc = 0;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(get(&f.os, "d/_delta_log/0.json", &rc), "first");
        EXPECT_EQ(rc, kOsOk);
    }
    EXPECT_EQ(f.fake.mints.load(), 1);
    EXPECT_EQ(f.store()->token_mints, 1u);
    EXPECT_EQ(f.fake.storage_unauth.load(), 0);
    EXPECT_EQ(f.fake.storage_requests.load(), 7);
}

TEST(BoltGcsOAuth, RemintsWithinRefreshMarginOnly) {
    Fixture f;
    ASSERT_TRUE(f.open());
    ASSERT_EQ(put(&f.os, "k", "v", false), kOsOk);
    ASSERT_EQ(f.fake.mints.load(), 1);
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    f.store()->bearer_expiry_unix = now + gcs::kGcsRefreshMarginS + 30u;
    ASSERT_EQ(put(&f.os, "k", "v", false), kOsOk);
    EXPECT_EQ(f.fake.mints.load(), 1);
    f.store()->bearer_expiry_unix = now + gcs::kGcsRefreshMarginS - 1u;
    ASSERT_EQ(put(&f.os, "k", "v", false), kOsOk);
    EXPECT_EQ(f.fake.mints.load(), 2);
    EXPECT_GT(f.store()->bearer_expiry_unix, now + 3000u);
    EXPECT_STREQ(f.store()->bearer_token, "ya29.fake-2");
    EXPECT_EQ(f.fake.storage_unauth.load(), 0);
}

TEST(BoltGcsOAuth, ShortLivedTokensAreReMintedEveryOp) {
    Fixture f;
    f.fake.expires_in = 30;   // already inside the 60s margin
    ASSERT_TRUE(f.open());
    for (int i = 0; i < 3; ++i) ASSERT_EQ(put(&f.os, "k", "v", false), kOsOk);
    EXPECT_EQ(f.fake.mints.load(), 3);
}

TEST(BoltGcsOAuth, ConcurrentColdStartMintsOnce) {
    Fixture f;
    ASSERT_TRUE(f.open());
    constexpr int kThreads = 8;
    std::atomic<int> ok{0};
    std::thread th[kThreads];
    for (int t = 0; t < kThreads; ++t) {
        th[t] = std::thread([&, t] {
            if (put(&f.os, "c/" + std::to_string(t), "x", true) == kOsOk) {
                ok.fetch_add(1);
            }
        });
    }
    for (auto& t : th) t.join();
    EXPECT_EQ(ok.load(), kThreads);
    EXPECT_EQ(f.fake.mints.load(), 1);
    EXPECT_EQ(f.fake.storage_unauth.load(), 0);
}

// A refused exchange must fail the op, never fall back to anonymous.
TEST(BoltGcsOAuth, RefusedExchangeFailsClosed) {
    Fixture f;
    f.fake.refuse = true;
    ASSERT_TRUE(f.open());
    EXPECT_EQ(put(&f.os, "k", "v", true), kOsIoError);
    int rc = 0;
    (void)get(&f.os, "k", &rc);
    EXPECT_EQ(rc, kOsIoError);
    EXPECT_EQ(f.fake.storage_requests.load(), 0);
    EXPECT_NE(std::strstr(f.store()->last_auth_error, "invalid_grant"),
              nullptr);
}

// The fake really verifies: a JWT signed by a different key is refused.
TEST(BoltGcsOAuth, SignatureFromAnotherKeyIsRejected) {
    Fixture f;
    EVP_PKEY* other = new_rsa();
    ASSERT_NE(other, nullptr);
    f.fake.verify_key = other;
    ASSERT_TRUE(f.open());
    EXPECT_EQ(put(&f.os, "k", "v", false), kOsIoError);
    EXPECT_EQ(f.fake.last_jwt_error, "signature");
    EXPECT_EQ(f.fake.mints.load(), 0);
    EVP_PKEY_free(other);
}

TEST(BoltGcsOAuth, TokenWithControlCharsIsNotSent) {
    Fixture f;
    f.fake.token_override = "ya29.bad token";
    ASSERT_TRUE(f.open());
    EXPECT_EQ(put(&f.os, "k", "v", false), kOsIoError);
    EXPECT_EQ(f.fake.storage_requests.load(), 0);
}

TEST(BoltGcsOAuth, TokenUriDefaultsToGoogleAndRejectsOtherSchemes) {
    EVP_PKEY* key = new_rsa();
    ASSERT_NE(key, nullptr);
    const std::string pem = pem_of(key);
    EVP_PKEY_free(key);
    bolt::Arena arena;
    static gcs::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.auth_mode = gcs::AuthMode::ServiceAccount;
    std::string j = sa_json(pem, "");
    std::memcpy(cfg.service_account_json.bytes, j.data(), j.size());
    cfg.service_account_json.len = static_cast<uint32_t>(j.size());
    ObjectStore os{};
    ASSERT_TRUE(gcs::gcs_store_new(&os, &arena, &cfg));
    EXPECT_STREQ(static_cast<gcs::GcsStore*>(os.impl)->token_uri,
                 "https://oauth2.googleapis.com/token");
    j = sa_json(pem, "ftp://example.com/token");
    std::memcpy(cfg.service_account_json.bytes, j.data(), j.size());
    cfg.service_account_json.len = static_cast<uint32_t>(j.size());
    EXPECT_FALSE(gcs::gcs_store_new(&os, &arena, &cfg));
}

// HMAC put_if_absent over the XML API carries a signed
// x-goog-if-generation-match: 0, and the 412 maps to kOsExists.
TEST(BoltGcsOAuth, HmacXmlPutIfAbsentIsGenerationMatchCas) {
    FakeGoogle fake;
    ASSERT_TRUE(fake.start());
    bolt::Arena arena;
    static gcs::Config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    std::strncpy(cfg.bucket, "bkt", sizeof(cfg.bucket) - 1u);
    cfg.auth_mode = gcs::AuthMode::Hmac;
    const std::string ep = fake.endpoint();
    std::strncpy(cfg.endpoint_override, ep.c_str(),
                 sizeof(cfg.endpoint_override) - 1u);
    ASSERT_TRUE(s3_compat::secret_set(&cfg.hmac_access_id, "GOOG1EXAMPLE"));
    ASSERT_TRUE(s3_compat::secret_set(&cfg.hmac_secret, "GOOGSECRET"));
    ObjectStore os{};
    ASSERT_TRUE(gcs::gcs_store_new(&os, &arena, &cfg));
    EXPECT_EQ(put(&os, "d/_delta_log/0.json", "first", true), kOsOk);
    EXPECT_EQ(put(&os, "d/_delta_log/0.json", "second", true), kOsExists);
    int rc = 0;
    EXPECT_EQ(get(&os, "d/_delta_log/0.json", &rc), "first");
    EXPECT_EQ(put(&os, "d/_delta_log/0.json", "third", false), kOsOk);
    EXPECT_EQ(get(&os, "d/_delta_log/0.json", &rc), "third");
    EXPECT_EQ(fake.xml_cas_puts.load(), 2);
    EXPECT_EQ(fake.xml_unsigned_cas.load(), 0);
    EXPECT_EQ(fake.mints.load(), 0);
}

}  // namespace
#else
TEST(BoltGcsOAuth, NeedsTlsAndPosix) { GTEST_SKIP() << "no OpenSSL / POSIX"; }
#endif
