// sha1.h — SHA-1 (FIPS 180-4). For integrity checks that name it (S3
// x-amz-checksum-sha1, git); never for anything security-bearing.

#pragma once

#include <cstdint>

namespace bolt {
namespace crypto {

void sha1(const uint8_t* data, uint64_t len, uint8_t out[20]) noexcept;

}  // namespace crypto
}  // namespace bolt
