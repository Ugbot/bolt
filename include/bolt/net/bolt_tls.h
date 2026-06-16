// bolt/net/bolt_tls.h — minimal OpenSSL 3.x TLS 1.3 client wrapper.
//
// Scope: client-side only. TLS 1.3 only (no 1.2 fallback). PODs only.
// All OpenSSL types live behind `void*` so callers can't poke at them —
// keeps this header free of any OpenSSL include.
//
// Tiger Style:
//   - No exceptions, no RTTI, no smart pointers.
//   - Bounded fields (fixed char arrays for paths / SNI).
//   - Two-phase init: caller stack-allocates TlsSocket, calls connect.
//   - Errors signalled via bool / negative return — no throws.
//
// Build-time toggle:
//   - When `BOLT_WITH_TLS` is undefined (no OpenSSL at configure time),
//     the .cpp ships stub bodies that return failure so consumers still
//     link.
#pragma once

#include <cstdint>

namespace bolt::net {

// Caller-owned configuration. Plain POD; copy/move freely.
//
// `server_name`         — required for client SNI + cert hostname check;
//                         null-terminated, ≤255 chars.
// `ca_path`             — PEM file or hashed-dir of trust roots; empty
//                         string ("") = fall back to OpenSSL's default
//                         system trust store (`SSL_CTX_set_default_verify_paths`).
// `client_cert_path` /
// `client_key_path`     — mTLS client cert + private key (PEM). Both
//                         empty = no client cert.
// `verify_peer`         — true (default) → SSL_VERIFY_PEER + cert+hostname
//                         check. Set false ONLY for diagnostics.
// `insecure_skip_verify`— honoured only when verify_peer = false; lets
//                         the handshake proceed against self-signed certs.
// `handshake_timeout_ms`— wall-clock deadline for the connect+handshake
//                         loop. 0 = library default (5000 ms).
// `recv_timeout_ms`     — per-call deadline for send/recv. 0 = library
//                         default (5000 ms).
struct TlsConfig {
    char           server_name[256];
    char           ca_path[512];
    char           client_cert_path[512];
    char           client_key_path[512];
    bool           verify_peer;
    bool           insecure_skip_verify;
    std::uint16_t  handshake_timeout_ms;
    std::uint16_t  recv_timeout_ms;
};

// Live connection handle. Owned by caller; pass by pointer to all funcs.
// `ssl_ctx` / `ssl` are opaque (`SSL_CTX*` / `SSL*` internally).
// `fd` is the underlying TCP socket (SOCKET on Windows, int on POSIX —
// stored as int either way; cast at the call site in the .cpp).
// `handshake_done` flips to true after a successful tls_client_connect.
struct TlsSocket {
    void*  ssl_ctx;
    void*  ssl;
    int    fd;
    bool   handshake_done;
};

// Connect a fresh TCP socket to (ipv4, port), then drive an OpenSSL 3.x
// TLS 1.3 handshake on it. `ipv4` is in host byte order.
// Returns true iff the handshake completes and (when verify_peer is set)
// the peer cert + hostname check pass.
// On failure, any partially-opened resources are torn down before return.
bool tls_client_connect(TlsSocket*       out,
                        const TlsConfig* cfg,
                        std::uint32_t    ipv4,
                        std::uint16_t    port) noexcept;

// Write `len` bytes. Returns `len` on success, -1 on error.
// Bounded by cfg.recv_timeout_ms (per WANT_READ/WANT_WRITE re-arm).
std::int64_t tls_send_all(const TlsSocket* s,
                          const void*      data,
                          std::uint32_t    len) noexcept;

// Read exactly `len` bytes. Returns true on full read, false on EOF /
// error / timeout.
bool tls_recv_all(const TlsSocket* s,
                  void*            data,
                  std::uint32_t    len) noexcept;

// Read up to `cap` bytes. Returns bytes read (>=0), or -1 on error.
// 0 means clean peer shutdown.
std::int32_t tls_recv_some(const TlsSocket* s,
                           void*            data,
                           std::uint32_t    cap) noexcept;

// Tear down: SSL_shutdown → SSL_free → SSL_CTX_free → close fd.
// Safe to call on a zeroed TlsSocket (no-op).
void tls_close(TlsSocket* s) noexcept;

}  // namespace bolt::net
