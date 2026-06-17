// Ed25519 keygen / detached sign / verify over OpenSSL 3.x EVP.
// See include/bolt/crypto/ed25519.h for the contract. The curve math is
// OpenSSL (EVP_PKEY_ED25519); only the raw-key plumbing lives here.
//
// Without BOLT_WITH_TLS every entry point is an explicit failure stub
// (returns false) — same convention as noise.cpp.

#include "bolt/crypto/ed25519.h"

#include <cassert>
#include <cstring>

#if defined(BOLT_WITH_TLS)
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

namespace bolt {
namespace crypto {

#if defined(BOLT_WITH_TLS)
namespace {

// Derive the 32-byte raw public key for a private key built from `secret`.
// Returns true on success. The EVP_PKEY is created and freed internally.
bool derive_pub_from_secret(const std::uint8_t secret[k_ed25519_sec_len],
                            std::uint8_t pub[k_ed25519_pub_len]) noexcept {
    assert(secret != nullptr);
    assert(pub != nullptr);
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519, nullptr, secret, k_ed25519_sec_len);
    if (pkey == nullptr) return false;
    std::size_t pub_len = k_ed25519_pub_len;
    const int ok = EVP_PKEY_get_raw_public_key(pkey, pub, &pub_len);
    EVP_PKEY_free(pkey);
    assert(!(ok == 1) || pub_len == k_ed25519_pub_len);
    return ok == 1 && pub_len == k_ed25519_pub_len;
}

}  // namespace

bool ed25519_keypair_generate(Ed25519Keypair* kp) noexcept {
    assert(kp != nullptr);
    std::uint8_t seed[k_ed25519_sec_len];
    if (RAND_bytes(seed, static_cast<int>(k_ed25519_sec_len)) != 1) {
        return false;
    }
    if (!ed25519_keypair_from_secret(kp, seed)) {
        std::memset(seed, 0, sizeof(seed));
        return false;
    }
    std::memset(seed, 0, sizeof(seed));
    // Postcondition: a generated public key is never all-zero.
    bool all_zero = true;
    for (std::uint32_t i = 0; i < k_ed25519_pub_len; ++i) {
        if (kp->pub[i] != 0u) { all_zero = false; break; }
    }
    return !all_zero;
}

bool ed25519_keypair_from_secret(
    Ed25519Keypair* kp,
    const std::uint8_t secret[k_ed25519_sec_len]) noexcept {
    assert(kp != nullptr);
    assert(secret != nullptr);
    std::memcpy(kp->secret, secret, k_ed25519_sec_len);
    return derive_pub_from_secret(kp->secret, kp->pub);
}

bool ed25519_sign(const Ed25519Keypair* kp,
                  const std::uint8_t* msg, std::uint32_t msg_len,
                  std::uint8_t sig[k_ed25519_sig_len]) noexcept {
    assert(kp != nullptr);
    assert(sig != nullptr);
    assert(msg != nullptr || msg_len == 0u);
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519, nullptr, kp->secret, k_ed25519_sec_len);
    if (pkey == nullptr) return false;
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (mdctx == nullptr) { EVP_PKEY_free(pkey); return false; }
    bool ok = false;
    // Ed25519 is one-shot: DigestSignInit with a null MD, then DigestSign.
    std::size_t sig_len = k_ed25519_sig_len;
    if (EVP_DigestSignInit(mdctx, nullptr, nullptr, nullptr, pkey) == 1 &&
        EVP_DigestSign(mdctx, sig, &sig_len, msg,
                       static_cast<std::size_t>(msg_len)) == 1) {
        ok = (sig_len == k_ed25519_sig_len);
    }
    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    assert(!ok || sig_len == k_ed25519_sig_len);
    return ok;
}

bool ed25519_verify(const std::uint8_t pub[k_ed25519_pub_len],
                    const std::uint8_t* msg, std::uint32_t msg_len,
                    const std::uint8_t sig[k_ed25519_sig_len]) noexcept {
    assert(pub != nullptr);
    assert(sig != nullptr);
    assert(msg != nullptr || msg_len == 0u);
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, nullptr, pub, k_ed25519_pub_len);
    if (pkey == nullptr) return false;
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (mdctx == nullptr) { EVP_PKEY_free(pkey); return false; }
    bool ok = false;
    if (EVP_DigestVerifyInit(mdctx, nullptr, nullptr, nullptr, pkey) == 1) {
        ok = EVP_DigestVerify(mdctx, sig, k_ed25519_sig_len, msg,
                              static_cast<std::size_t>(msg_len)) == 1;
    }
    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    return ok;
}

#else  // !BOLT_WITH_TLS — explicit failure stubs.

bool ed25519_keypair_generate(Ed25519Keypair* kp) noexcept {
    assert(kp != nullptr);
    (void)kp;
    return false;
}

bool ed25519_keypair_from_secret(
    Ed25519Keypair* kp,
    const std::uint8_t secret[k_ed25519_sec_len]) noexcept {
    assert(kp != nullptr);
    assert(secret != nullptr);
    (void)kp;
    (void)secret;
    return false;
}

bool ed25519_sign(const Ed25519Keypair* kp,
                  const std::uint8_t* msg, std::uint32_t msg_len,
                  std::uint8_t sig[k_ed25519_sig_len]) noexcept {
    assert(kp != nullptr);
    assert(sig != nullptr);
    (void)kp;
    (void)msg;
    (void)msg_len;
    (void)sig;
    return false;
}

bool ed25519_verify(const std::uint8_t pub[k_ed25519_pub_len],
                    const std::uint8_t* msg, std::uint32_t msg_len,
                    const std::uint8_t sig[k_ed25519_sig_len]) noexcept {
    assert(pub != nullptr);
    assert(sig != nullptr);
    (void)pub;
    (void)msg;
    (void)msg_len;
    (void)sig;
    return false;
}

#endif  // BOLT_WITH_TLS

}  // namespace crypto
}  // namespace bolt
