// bolt/lakehouse/catalog/secret.h — bounded credential POD shared across
// catalog backends. Caller writes a NUL-terminated string into `value`;
// consumers treat it as opaque. Tiger Style: POD, fixed cap, no exceptions.
#pragma once

#include <cstdint>

namespace bolt {
namespace lakehouse {

static constexpr uint32_t kSecretMax = 256u;

struct Secret {
    char value[kSecretMax];
};

}  // namespace lakehouse
}  // namespace bolt
