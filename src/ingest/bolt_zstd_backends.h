// src/ingest/bolt_zstd_backends.h (PRIVATE) — the two implementations behind bolt/ingest/bolt_zstd.h.
// bolt_zstd.cpp picks one per direction at build time; tests/test_bolt_zstd_backends.cpp runs both.
#pragma once

#include <cstdint>

namespace bolt {
namespace ingest {
namespace zstd_backend {

// bolt's own (always built).
uint64_t self_bound(uint64_t src_len) noexcept;
int self_compress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t* dst_len, int level) noexcept;
uint64_t self_scratch_size() noexcept;
int self_decode(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
                void* scratch, uint64_t scratch_len) noexcept;

// libzstd (built when BOLT_HAS_ZSTD).
bool ref_built() noexcept;
uint64_t ref_bound(uint64_t src_len) noexcept;
int ref_compress(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t* dst_len, int level) noexcept;
uint64_t ref_scratch_size() noexcept;
int ref_decode(const uint8_t* src, uint64_t src_len, uint8_t* dst, uint64_t dst_cap, uint64_t* out_len,
               void* scratch, uint64_t scratch_len) noexcept;

}  // namespace zstd_backend
}  // namespace ingest
}  // namespace bolt
