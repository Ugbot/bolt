// bolt/crypto/ed25519.h — Ed25519 keypair generation, detached signing, and
// verification over OpenSSL 3.x EVP (EVP_PKEY_ED25519). The curve math is
// OpenSSL's vetted implementation — never hand-rolled (see noise.h's crypto
// policy). This wrapper is the owned primitive consumers (chukonu's cluster
// ACL, G2CLUS-43) reach for instead of scattering EVP calls.
//
// All keys/signatures are raw bytes (32-byte public key, 32-byte seed/secret,
// 64-byte signature) — no PEM, no DER, no heap. The signature is "detached":
// it covers the message bytes directly (Ed25519 is single-shot, no pre-hash).
//
// When BOLT_WITH_TLS is not defined, every function compiles to an explicit
// failure stub (returns false) — same convention as noise.cpp.
//
// Tiger Style: trivially-copyable PODs, fixed-size buffers (no allocation),
// explicit widths, no exceptions (bool returns), >=2 asserts/fn.

#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace crypto {

// ---------------------------------------------------------------------------
// Sizes (all compile-time; nothing here allocates).
// ---------------------------------------------------------------------------
constexpr std::uint32_t k_ed25519_pub_len = 32u;   // raw public key
constexpr std::uint32_t k_ed25519_sec_len = 32u;   // raw private seed
constexpr std::uint32_t k_ed25519_sig_len = 64u;   // detached signature

// ---------------------------------------------------------------------------
// Ed25519 keypair (POD). `secret` is the 32-byte seed; `pub` the public key
// derived from it. Both are raw little-endian bytes per RFC 8032.
// ---------------------------------------------------------------------------
struct Ed25519Keypair {
    std::uint8_t secret[k_ed25519_sec_len];
    std::uint8_t pub[k_ed25519_pub_len];
};
static_assert(sizeof(Ed25519Keypair) == 64, "Ed25519Keypair is 64 bytes");

// Generate a fresh Ed25519 keypair via OpenSSL. Returns false on RNG/OpenSSL
// failure. Never returns an all-zero key on success.
bool ed25519_keypair_generate(Ed25519Keypair* kp) noexcept;

// Derive the public key from an existing 32-byte seed (for fixed identities
// in tests / on-disk node keys). Returns false on OpenSSL failure.
bool ed25519_keypair_from_secret(
    Ed25519Keypair* kp,
    const std::uint8_t secret[k_ed25519_sec_len]) noexcept;

// Sign `msg` with the keypair's secret. Writes a 64-byte detached signature
// into `sig`. Returns false on OpenSSL failure. `msg` may be null iff
// msg_len == 0.
bool ed25519_sign(const Ed25519Keypair* kp,
                  const std::uint8_t* msg, std::uint32_t msg_len,
                  std::uint8_t sig[k_ed25519_sig_len]) noexcept;

// Verify `sig` over `msg` against the raw 32-byte public key `pub`. Returns
// true iff the signature is valid. A tampered message, forged signature, or
// wrong key all return false — the standard rejection path. `msg` may be null
// iff msg_len == 0.
bool ed25519_verify(const std::uint8_t pub[k_ed25519_pub_len],
                    const std::uint8_t* msg, std::uint32_t msg_len,
                    const std::uint8_t sig[k_ed25519_sig_len]) noexcept;

}  // namespace crypto
}  // namespace bolt
