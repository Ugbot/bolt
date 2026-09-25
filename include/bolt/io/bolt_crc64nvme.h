// bolt_crc64nvme.h — CRC-64/NVME (the S3 x-amz-checksum-crc64nvme algorithm).
//
// Poly 0xAD93D23594C93659, reflected in/out, init and xorout all-ones:
// crc64nvme("123456789") == 0xAE8B14860A799888. Software slicing-by-8; the
// tables are constexpr in the .cpp, so there is no runtime init.
//
// seed is a prior result (0 for a fresh CRC), matching bolt::io::crc32c.

#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt {
namespace io {

uint64_t crc64nvme(const void* data, size_t len, uint64_t seed = 0) noexcept;

inline uint64_t crc64nvme_update(uint64_t crc, const void* data,
                                 size_t len) noexcept {
    return crc64nvme(data, len, crc);
}

}  // namespace io
}  // namespace bolt
