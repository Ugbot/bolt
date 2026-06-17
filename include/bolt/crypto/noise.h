// Noise XX handshake + ChaCha20-Poly1305 AEAD — the crypto primitive behind
// chukonu's encrypted transport (G2CLUS-30). Suite:
//
//   Noise_XX_25519_ChaChaPoly_SHA256
//
// The three XX messages:  -> e ; <- e, ee, s, es ; -> s, se
// After the final message both peers hold two CipherStates (one per
// direction) keyed off the shared chaining key — the per-stream session keys.
//
// Crypto policy: the elliptic-curve (X25519) and AEAD (ChaCha20-Poly1305)
// math is OpenSSL 3.x (EVP), a vetted implementation — never hand-rolled.
// SHA-256 / HMAC reuse bolt::crypto's self-contained implementations. Only
// the Noise *framing* (MixHash / MixKey / HKDF chaining) is composed here;
// that framing is protocol glue, not novel cryptography.
//
// Tiger Style: trivially-copyable PODs, fixed-size buffers (no allocation),
// explicit widths, no exceptions (bool/enum returns), >=2 asserts/fn.
// Nonce reuse is fatal and asserted against.

#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace crypto {

// ---------------------------------------------------------------------------
// Sizes (all compile-time; nothing here allocates).
// ---------------------------------------------------------------------------
constexpr std::uint32_t k_noise_dh_len   = 32u;  // X25519 public/shared len
constexpr std::uint32_t k_noise_hash_len = 32u;  // SHA-256
constexpr std::uint32_t k_noise_key_len  = 32u;  // ChaCha20-Poly1305 key
constexpr std::uint32_t k_noise_tag_len  = 16u;  // Poly1305 tag
constexpr std::uint32_t k_noise_nonce_len = 12u; // ChaCha20-Poly1305 nonce

// XX message buffer caps. Message 2 (<- e, ee, s, es) is the largest:
// 32 (e) + 32 (s ciphertext) + 16 (s tag) = 80 bytes of handshake material
// when a zero-length payload rides along; plus optional payload + its tag.
constexpr std::uint32_t k_noise_max_msg = 256u;

// ---------------------------------------------------------------------------
// X25519 keypair (POD). `secret` is the clamped scalar; `pub` the public key.
// ---------------------------------------------------------------------------
struct NoiseKeypair {
    std::uint8_t secret[k_noise_dh_len];
    std::uint8_t pub[k_noise_dh_len];
};
static_assert(sizeof(NoiseKeypair) == 64, "NoiseKeypair is 64 bytes");

// Generate a fresh X25519 keypair via OpenSSL. Returns false on RNG/OpenSSL
// failure. Never returns an all-zero key.
bool noise_keypair_generate(NoiseKeypair* kp) noexcept;

// Derive a public key from an existing 32-byte X25519 secret (for fixed
// static identities in tests). Returns false on OpenSSL failure.
bool noise_keypair_from_secret(NoiseKeypair* kp,
                               const std::uint8_t secret[k_noise_dh_len]) noexcept;

// ---------------------------------------------------------------------------
// CipherState — a keyed ChaCha20-Poly1305 stream with a monotonic 64-bit
// nonce counter. After a successful handshake each peer holds two of these
// (send / recv). This is the per-stream session key the transport AEADs with.
// ---------------------------------------------------------------------------
struct NoiseCipherState {
    std::uint8_t key[k_noise_key_len];
    std::uint64_t nonce;       // next counter to use; never reused
    bool          has_key;
};
static_assert(sizeof(NoiseCipherState) <= 48, "NoiseCipherState compact");

// AEAD encrypt with an EXPLICIT nonce counter (transport path: the counter is
// derived from (msg_seq, frag) so out-of-order datagrams decrypt independently
// — the CipherState's internal counter is NOT advanced). `ad` is additional
// authenticated data (may be null if ad_len==0). `out` must hold
// plaintext_len + k_noise_tag_len bytes. Returns false on OpenSSL failure.
bool noise_aead_encrypt(const NoiseCipherState* cs, std::uint64_t counter,
                        const std::uint8_t* ad, std::uint32_t ad_len,
                        const std::uint8_t* plaintext, std::uint32_t pt_len,
                        std::uint8_t* out) noexcept;

// AEAD decrypt the counterpart. `in` is ciphertext+tag (in_len includes the
// 16-byte tag). `out` holds in_len - k_noise_tag_len plaintext bytes. Returns
// false on tag-mismatch (tamper) or OpenSSL failure — the standard rejection
// path for a forged or corrupted fragment.
bool noise_aead_decrypt(const NoiseCipherState* cs, std::uint64_t counter,
                        const std::uint8_t* ad, std::uint32_t ad_len,
                        const std::uint8_t* in, std::uint32_t in_len,
                        std::uint8_t* out) noexcept;

// ---------------------------------------------------------------------------
// Handshake state machine (POD). One per peer per stream. The symmetric
// state (chaining key `ck`, transcript hash `h`, in-handshake cipher `cs`)
// plus the local ephemeral / static keys and the remote public keys.
// ---------------------------------------------------------------------------
struct NoiseHandshake {
    std::uint8_t  ck[k_noise_hash_len];   // chaining key
    std::uint8_t  h[k_noise_hash_len];    // transcript hash
    NoiseCipherState cs;                  // in-handshake AEAD (after first MixKey)

    NoiseKeypair  e;                      // local ephemeral
    NoiseKeypair  s;                      // local static
    std::uint8_t  re[k_noise_dh_len];     // remote ephemeral pub
    std::uint8_t  rs[k_noise_dh_len];     // remote static pub

    std::uint8_t  msg_index;              // which XX message comes next (0..2)
    bool          initiator;
    bool          have_re;
    bool          have_rs;
};

// Initialise a handshake. `static_kp` is this peer's long-term identity (XX
// authenticates both sides' statics). `initiator` true on the side that sends
// message 1 (-> e). Returns false on OpenSSL failure setting up the ephemeral.
bool noise_handshake_init(NoiseHandshake* hs, bool initiator,
                          const NoiseKeypair* static_kp) noexcept;

// Write the next handshake message into `out` (cap k_noise_max_msg). On the
// final message (msg_index reaches 3) the split CipherStates are produced via
// noise_handshake_split. `payload`/`payload_len` is optional in-handshake data
// (may be null/0). Returns the number of bytes written, or 0 on error / when
// it is not this peer's turn to write.
std::uint32_t noise_handshake_write(NoiseHandshake* hs,
                                    const std::uint8_t* payload,
                                    std::uint32_t payload_len,
                                    std::uint8_t* out,
                                    std::uint32_t out_cap) noexcept;

// Read a handshake message from `in`. Extracts any in-handshake payload into
// `payload_out` (cap payload_cap) and writes the recovered length to
// `*payload_len_out`. Returns false on a malformed message, AEAD tamper, or
// when it is not this peer's turn to read.
bool noise_handshake_read(NoiseHandshake* hs,
                          const std::uint8_t* in, std::uint32_t in_len,
                          std::uint8_t* payload_out, std::uint32_t payload_cap,
                          std::uint32_t* payload_len_out) noexcept;

// True once all three XX messages have been processed by this peer.
inline bool noise_handshake_done(const NoiseHandshake* hs) noexcept {
    return hs != nullptr && hs->msg_index >= 3u;
}

// Split the completed handshake into the two directional CipherStates.
// `send` is keyed for this peer's outbound traffic, `recv` for inbound. The
// initiator's send == the responder's recv (same key), so both peers derive
// the SAME pair of session keys (mirror-imaged). Returns false if the
// handshake is not yet complete.
bool noise_handshake_split(const NoiseHandshake* hs,
                           NoiseCipherState* send,
                           NoiseCipherState* recv) noexcept;

}  // namespace crypto
}  // namespace bolt
