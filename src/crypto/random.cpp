#include "bolt/crypto/random.h"

#include <cassert>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#elif defined(__APPLE__)
#include <sys/random.h>
#elif defined(__linux__)
#include <unistd.h>
#else
#error "bolt::entropy requires Windows, macOS, or Linux OS entropy"
#endif

namespace bolt::crypto {

bool random_bytes(uint8_t* output, uint32_t size) noexcept {
    if (output == nullptr || size == 0 || size > kRandomBytesCap) {
        return false;
    }
    assert(output != nullptr);
    assert(size > 0 && size <= kRandomBytesCap);
#if defined(_WIN32)
    const bool ok = BCryptGenRandom(nullptr, output, static_cast<ULONG>(size),
                                   BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    const bool ok = getentropy(output, size) == 0;
#endif
    if (!ok) {
        std::memset(output, 0, size);
        return false;
    }
    return true;
}

}  // namespace bolt::crypto
