// Noise XX (Noise_XX_25519_ChaChaPoly_SHA256) over OpenSSL 3.x EVP.
// See include/bolt/crypto/noise.h for the contract. X25519 + ChaCha20-
// Poly1305 are OpenSSL; SHA-256 / HMAC are bolt::crypto; the Noise framing
// (MixHash / MixKey / HKDF) is composed here per the Noise spec.

#include "bolt/crypto/noise.h"
#include "bolt/crypto/sigv4.h"   // bolt::crypto::sha256 / hmac_sha256

#include <cassert>
#include <cstring>

#if defined(BOLT_WITH_TLS)
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

namespace bolt {
namespace crypto {

namespace {

// Noise protocol name for the suite. Initial h = SHA256(name) since the name
// is exactly 32 bytes (the hash length), per the Noise spec.
constexpr char k_proto_name[] = "Noise_XX_25519_ChaChaPoly_SHA256";
static_assert(sizeof(k_proto_name) - 1u == k_noise_hash_len,
              "protocol name must be exactly one hash block");

// HKDF (Noise variant) from HMAC-SHA256: derive 2 or 3 outputs from
// (chaining_key, input_key_material). Each output is k_noise_hash_len bytes.
void hkdf2(const std::uint8_t ck[k_noise_hash_len],
           const std::uint8_t* ikm, std::uint32_t ikm_len,
           std::uint8_t out1[k_noise_hash_len],
           std::uint8_t out2[k_noise_hash_len]) noexcept {
    assert(ck != nullptr);
    assert(ikm != nullptr || ikm_len == 0u);
    std::uint8_t temp_key[k_noise_hash_len];
    hmac_sha256(ck, k_noise_hash_len, ikm, ikm_len, temp_key);
    const std::uint8_t one = 0x01u;
    hmac_sha256(temp_key, k_noise_hash_len, &one, 1u, out1);
    std::uint8_t in2[k_noise_hash_len + 1u];
    std::memcpy(in2, out1, k_noise_hash_len);
    in2[k_noise_hash_len] = 0x02u;
    hmac_sha256(temp_key, k_noise_hash_len, in2, k_noise_hash_len + 1u, out2);
}

// MixHash: h = SHA256(h || data).
void mix_hash(NoiseHandshake* hs, const std::uint8_t* data,
              std::uint32_t len) noexcept {
    assert(hs != nullptr);
    assert(data != nullptr || len == 0u);
    std::uint8_t buf[k_noise_hash_len + k_noise_max_msg];
    assert(len <= k_noise_max_msg);
    std::memcpy(buf, hs->h, k_noise_hash_len);
    std::memcpy(buf + k_noise_hash_len, data, len);
    sha256(buf, k_noise_hash_len + len, hs->h);
}

// MixKey: (ck, temp) = HKDF2(ck, dh); cipher key = temp.
void mix_key(NoiseHandshake* hs, const std::uint8_t* dh,
             std::uint32_t dh_len) noexcept {
    assert(hs != nullptr && dh != nullptr);
    assert(dh_len == k_noise_dh_len);
    std::uint8_t new_ck[k_noise_hash_len];
    std::uint8_t temp_k[k_noise_hash_len];
    hkdf2(hs->ck, dh, dh_len, new_ck, temp_k);
    std::memcpy(hs->ck, new_ck, k_noise_hash_len);
    std::memcpy(hs->cs.key, temp_k, k_noise_key_len);
    hs->cs.nonce = 0u;
    hs->cs.has_key = true;
}

#if defined(BOLT_WITH_TLS)
// X25519 shared secret via OpenSSL EVP. Returns false on any failure.
bool x25519(const std::uint8_t secret[k_noise_dh_len],
            const std::uint8_t peer_pub[k_noise_dh_len],
            std::uint8_t out[k_noise_dh_len]) noexcept {
    assert(secret != nullptr && peer_pub != nullptr && out != nullptr);
    EVP_PKEY* sk = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_X25519, nullptr, secret, k_noise_dh_len);
    EVP_PKEY* pk = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_X25519, nullptr, peer_pub, k_noise_dh_len);
    bool ok = false;
    if (sk != nullptr && pk != nullptr) {
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(sk, nullptr);
        if (ctx != nullptr) {
            std::size_t len = k_noise_dh_len;
            ok = EVP_PKEY_derive_init(ctx) == 1 &&
                 EVP_PKEY_derive_set_peer(ctx, pk) == 1 &&
                 EVP_PKEY_derive(ctx, out, &len) == 1 &&
                 len == k_noise_dh_len;
            EVP_PKEY_CTX_free(ctx);
        }
    }
    EVP_PKEY_free(sk);
    EVP_PKEY_free(pk);
    return ok;
}

// Encode the 12-byte ChaCha20-Poly1305 nonce: 4 zero bytes + LE64(counter),
// per the Noise spec.
void encode_nonce(std::uint64_t counter,
                  std::uint8_t out[k_noise_nonce_len]) noexcept {
    out[0] = out[1] = out[2] = out[3] = 0u;
    for (std::uint32_t i = 0; i < 8u; ++i) {
        out[4u + i] = static_cast<std::uint8_t>((counter >> (8u * i)) & 0xFFu);
    }
}
#endif  // BOLT_WITH_TLS

}  // namespace

// ---------------------------------------------------------------------------
// Keypairs
// ---------------------------------------------------------------------------
bool noise_keypair_from_secret(NoiseKeypair* kp,
                               const std::uint8_t secret[k_noise_dh_len]) noexcept {
    assert(kp != nullptr && secret != nullptr);
#if defined(BOLT_WITH_TLS)
    std::memcpy(kp->secret, secret, k_noise_dh_len);
    EVP_PKEY* sk = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_X25519, nullptr, secret, k_noise_dh_len);
    if (sk == nullptr) return false;
    std::size_t plen = k_noise_dh_len;
    const bool ok = EVP_PKEY_get_raw_public_key(sk, kp->pub, &plen) == 1 &&
                    plen == k_noise_dh_len;
    EVP_PKEY_free(sk);
    assert(ok);  // a valid X25519 secret always yields a public key
    return ok;
#else
    (void)kp; (void)secret;
    return false;
#endif
}

bool noise_keypair_generate(NoiseKeypair* kp) noexcept {
    assert(kp != nullptr);
#if defined(BOLT_WITH_TLS)
    std::uint8_t secret[k_noise_dh_len];
    if (RAND_bytes(secret, static_cast<int>(k_noise_dh_len)) != 1) return false;
    const bool ok = noise_keypair_from_secret(kp, secret);
    assert(ok);
    return ok;
#else
    (void)kp;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// AEAD
// ---------------------------------------------------------------------------
bool noise_aead_encrypt(const NoiseCipherState* cs, std::uint64_t counter,
                        const std::uint8_t* ad, std::uint32_t ad_len,
                        const std::uint8_t* plaintext, std::uint32_t pt_len,
                        std::uint8_t* out) noexcept {
    assert(cs != nullptr && cs->has_key && out != nullptr);
    assert(plaintext != nullptr || pt_len == 0u);
#if defined(BOLT_WITH_TLS)
    std::uint8_t nonce[k_noise_nonce_len];
    encode_nonce(counter, nonce);
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (ctx == nullptr) return false;
    int outl = 0;
    bool ok = EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), nullptr,
                                 cs->key, nonce) == 1;
    if (ok && ad_len > 0u) {
        ok = EVP_EncryptUpdate(ctx, nullptr, &outl, ad,
                               static_cast<int>(ad_len)) == 1;
    }
    if (ok) {
        ok = EVP_EncryptUpdate(ctx, out, &outl, plaintext,
                               static_cast<int>(pt_len)) == 1;
    }
    int finl = 0;
    if (ok) ok = EVP_EncryptFinal_ex(ctx, out + outl, &finl) == 1;
    if (ok) {
        ok = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, k_noise_tag_len,
                                 out + pt_len) == 1;
    }
    EVP_CIPHER_CTX_free(ctx);
    return ok;
#else
    (void)cs; (void)counter; (void)ad; (void)ad_len;
    (void)plaintext; (void)pt_len; (void)out;
    return false;
#endif
}

bool noise_aead_decrypt(const NoiseCipherState* cs, std::uint64_t counter,
                        const std::uint8_t* ad, std::uint32_t ad_len,
                        const std::uint8_t* in, std::uint32_t in_len,
                        std::uint8_t* out) noexcept {
    assert(cs != nullptr && cs->has_key && in != nullptr);
    assert(in_len >= k_noise_tag_len);
#if defined(BOLT_WITH_TLS)
    const std::uint32_t ct_len = in_len - k_noise_tag_len;
    std::uint8_t nonce[k_noise_nonce_len];
    encode_nonce(counter, nonce);
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (ctx == nullptr) return false;
    int outl = 0;
    bool ok = EVP_DecryptInit_ex(ctx, EVP_chacha20_poly1305(), nullptr,
                                 cs->key, nonce) == 1;
    if (ok && ad_len > 0u) {
        ok = EVP_DecryptUpdate(ctx, nullptr, &outl, ad,
                               static_cast<int>(ad_len)) == 1;
    }
    if (ok) {
        ok = EVP_DecryptUpdate(ctx, out, &outl, in,
                               static_cast<int>(ct_len)) == 1;
    }
    if (ok) {
        // Set expected tag, then Final returns 0 on mismatch (tamper).
        std::uint8_t tag[k_noise_tag_len];
        std::memcpy(tag, in + ct_len, k_noise_tag_len);
        ok = EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, k_noise_tag_len,
                                 tag) == 1;
    }
    int finl = 0;
    if (ok) ok = EVP_DecryptFinal_ex(ctx, out + outl, &finl) == 1;
    EVP_CIPHER_CTX_free(ctx);
    return ok;
#else
    (void)cs; (void)counter; (void)ad; (void)ad_len;
    (void)in; (void)in_len; (void)out;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------
bool noise_handshake_init(NoiseHandshake* hs, bool initiator,
                          const NoiseKeypair* static_kp) noexcept {
    assert(hs != nullptr && static_kp != nullptr);
    std::memset(hs, 0, sizeof(*hs));
    // h = ck = SHA256(protocol_name) (name is exactly hash-length).
    std::memcpy(hs->h, k_proto_name, k_noise_hash_len);
    std::memcpy(hs->ck, hs->h, k_noise_hash_len);
    hs->cs.has_key = false;
    hs->cs.nonce = 0u;
    hs->s = *static_kp;
    hs->initiator = initiator;
    hs->msg_index = 0u;
    // XX has no pre-message public keys; prologue is empty → no MixHash here.
    if (!noise_keypair_generate(&hs->e)) return false;
    assert(hs->e.pub[0] != 0u || hs->e.pub[31] != 0u);
    return true;
}

// EncryptAndHash: AEAD-seal `pt` with the in-handshake cipher (counter from
// cs.nonce, then advance), MixHash the ciphertext. Returns bytes written.
static std::uint32_t encrypt_and_hash(NoiseHandshake* hs,
                                      const std::uint8_t* pt,
                                      std::uint32_t pt_len,
                                      std::uint8_t* out) noexcept {
    assert(hs != nullptr && out != nullptr);
    assert(pt != nullptr || pt_len == 0u);
    if (!hs->cs.has_key) {        // no key yet → plaintext, just MixHash
        if (pt_len > 0u) std::memcpy(out, pt, pt_len);
        mix_hash(hs, out, pt_len);
        return pt_len;
    }
    const std::uint64_t counter = hs->cs.nonce;
    if (!noise_aead_encrypt(&hs->cs, counter, hs->h, k_noise_hash_len,
                            pt, pt_len, out)) {
        return 0u;
    }
    ++hs->cs.nonce;             // never reuse a handshake nonce
    const std::uint32_t n = pt_len + k_noise_tag_len;
    mix_hash(hs, out, n);
    return n;
}

// DecryptAndHash: inverse of encrypt_and_hash. MixHash uses the CIPHERTEXT
// (pre-decrypt) so both peers hash identical bytes. Returns false on tamper.
static bool decrypt_and_hash(NoiseHandshake* hs, const std::uint8_t* in,
                             std::uint32_t in_len, std::uint8_t* out,
                             std::uint32_t* out_len) noexcept {
    assert(hs != nullptr && in != nullptr && out_len != nullptr);
    if (!hs->cs.has_key) {
        if (in_len > 0u) std::memcpy(out, in, in_len);
        mix_hash(hs, in, in_len);
        *out_len = in_len;
        return true;
    }
    assert(in_len >= k_noise_tag_len);
    const std::uint64_t counter = hs->cs.nonce;
    std::uint8_t ct_copy[k_noise_max_msg];
    assert(in_len <= k_noise_max_msg);
    std::memcpy(ct_copy, in, in_len);     // MixHash needs the ciphertext
    if (!noise_aead_decrypt(&hs->cs, counter, hs->h, k_noise_hash_len,
                            in, in_len, out)) {
        return false;
    }
    ++hs->cs.nonce;
    mix_hash(hs, ct_copy, in_len);
    *out_len = in_len - k_noise_tag_len;
    return true;
}

// DH helper: mix_key(local_secret · remote_pub).
static bool dh_mix(NoiseHandshake* hs, const std::uint8_t* secret,
                   const std::uint8_t* remote_pub) noexcept {
#if defined(BOLT_WITH_TLS)
    std::uint8_t shared[k_noise_dh_len];
    if (!x25519(secret, remote_pub, shared)) return false;
    mix_key(hs, shared, k_noise_dh_len);
    return true;
#else
    (void)hs; (void)secret; (void)remote_pub;
    return false;
#endif
}

std::uint32_t noise_handshake_write(NoiseHandshake* hs,
                                    const std::uint8_t* payload,
                                    std::uint32_t payload_len,
                                    std::uint8_t* out,
                                    std::uint32_t out_cap) noexcept {
    assert(hs != nullptr && out != nullptr);
    assert(out_cap >= k_noise_max_msg);  // caller must size for the largest msg
    (void)out_cap;                       // asserted in debug; silence Release
    const bool my_turn = hs->initiator ? (hs->msg_index % 2u == 0u)
                                       : (hs->msg_index % 2u == 1u);
    if (!my_turn || hs->msg_index >= 3u) return 0u;
    std::uint32_t off = 0u;
    if (hs->msg_index == 0u) {                    // -> e
        std::memcpy(out, hs->e.pub, k_noise_dh_len);
        mix_hash(hs, hs->e.pub, k_noise_dh_len);
        off = k_noise_dh_len;
    } else if (hs->msg_index == 1u) {             // <- e, ee, s, es
        std::memcpy(out, hs->e.pub, k_noise_dh_len);
        mix_hash(hs, hs->e.pub, k_noise_dh_len);
        off = k_noise_dh_len;
        if (!dh_mix(hs, hs->e.secret, hs->re)) return 0u;   // ee
        const std::uint32_t n = encrypt_and_hash(hs, hs->s.pub,
                                                  k_noise_dh_len, out + off);
        if (n == 0u) return 0u;
        off += n;
        if (!dh_mix(hs, hs->s.secret, hs->re)) return 0u;   // es
    } else {                                      // -> s, se
        const std::uint32_t n = encrypt_and_hash(hs, hs->s.pub,
                                                  k_noise_dh_len, out + off);
        if (n == 0u) return 0u;
        off += n;
        // se = DH(initiator static, responder ephemeral): our static secret
        // against the remote ephemeral (NOT our ephemeral / remote static).
        if (!dh_mix(hs, hs->s.secret, hs->re)) return 0u;   // se
    }
    const std::uint32_t pn = encrypt_and_hash(hs, payload, payload_len,
                                              out + off);
    if (payload_len > 0u && pn == 0u) return 0u;
    off += pn;
    assert(off <= k_noise_max_msg);
    ++hs->msg_index;
    return off;
}

bool noise_handshake_read(NoiseHandshake* hs,
                          const std::uint8_t* in, std::uint32_t in_len,
                          std::uint8_t* payload_out, std::uint32_t payload_cap,
                          std::uint32_t* payload_len_out) noexcept {
    assert(hs != nullptr && in != nullptr && payload_len_out != nullptr);
    const bool my_turn = hs->initiator ? (hs->msg_index % 2u == 1u)
                                       : (hs->msg_index % 2u == 0u);
    if (!my_turn || hs->msg_index >= 3u) return false;
    std::uint32_t off = 0u;
    if (hs->msg_index == 0u) {                    // -> e
        if (in_len < k_noise_dh_len) return false;
        std::memcpy(hs->re, in, k_noise_dh_len);
        hs->have_re = true;
        mix_hash(hs, hs->re, k_noise_dh_len);
        off = k_noise_dh_len;
    } else if (hs->msg_index == 1u) {             // <- e, ee, s, es
        if (in_len < k_noise_dh_len) return false;
        std::memcpy(hs->re, in, k_noise_dh_len);
        hs->have_re = true;
        mix_hash(hs, hs->re, k_noise_dh_len);
        off = k_noise_dh_len;
        if (!dh_mix(hs, hs->e.secret, hs->re)) return false;        // ee
        std::uint32_t sl = 0u;                                      // s
        if (in_len < off + k_noise_dh_len + k_noise_tag_len) return false;
        if (!decrypt_and_hash(hs, in + off, k_noise_dh_len + k_noise_tag_len,
                              hs->rs, &sl)) return false;
        hs->have_rs = true;
        off += k_noise_dh_len + k_noise_tag_len;
        if (!dh_mix(hs, hs->e.secret, hs->rs)) return false;        // es
    } else {                                      // -> s, se
        std::uint32_t sl = 0u;
        if (in_len < off + k_noise_dh_len + k_noise_tag_len) return false;
        if (!decrypt_and_hash(hs, in + off, k_noise_dh_len + k_noise_tag_len,
                              hs->rs, &sl)) return false;
        hs->have_rs = true;
        off += k_noise_dh_len + k_noise_tag_len;
        if (!dh_mix(hs, hs->e.secret, hs->rs)) return false;        // se
    }
    // Remaining bytes are the (possibly empty) payload.
    std::uint8_t scratch[k_noise_max_msg];
    std::uint32_t pl = 0u;
    const std::uint32_t rem = in_len - off;
    if (!decrypt_and_hash(hs, in + off, rem, scratch, &pl)) return false;
    if (pl > payload_cap) return false;
    if (pl > 0u) std::memcpy(payload_out, scratch, pl);
    *payload_len_out = pl;
    ++hs->msg_index;
    return true;
}

bool noise_handshake_split(const NoiseHandshake* hs, NoiseCipherState* send,
                           NoiseCipherState* recv) noexcept {
    assert(hs != nullptr && send != nullptr && recv != nullptr);
    if (hs->msg_index < 3u) return false;
    std::uint8_t k1[k_noise_hash_len];
    std::uint8_t k2[k_noise_hash_len];
    hkdf2(hs->ck, nullptr, 0u, k1, k2);   // zero-length IKM per Noise Split()
    // Initiator: send=k1, recv=k2. Responder mirrors so the keys line up.
    const std::uint8_t* sk = hs->initiator ? k1 : k2;
    const std::uint8_t* rk = hs->initiator ? k2 : k1;
    std::memset(send, 0, sizeof(*send));
    std::memset(recv, 0, sizeof(*recv));
    std::memcpy(send->key, sk, k_noise_key_len);
    std::memcpy(recv->key, rk, k_noise_key_len);
    send->has_key = recv->has_key = true;
    send->nonce = recv->nonce = 0u;
    return true;
}

}  // namespace crypto
}  // namespace bolt
