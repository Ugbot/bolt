#pragma once

#include <cstdint>

namespace bolt::api::core::limits {

// Kind B: fixed startup allocation bounds for listener retirement admission
// and readiness fd lookup. Changing either requires rebuilding engines and
// consumers together. Queues/tables never grow during retirement or dispatch;
// capacity refusal retains ownership, and unsupported readiness fds refuse.
inline constexpr uint32_t kAcceptRetirementCapacity = 4096u;
inline constexpr int kAcceptReadinessFdLimit = 65536;

static_assert(kAcceptRetirementCapacity == 4096u && kAcceptRetirementCapacity > 0u);
static_assert(kAcceptReadinessFdLimit == 65536 && kAcceptReadinessFdLimit <= INT32_MAX);

} // namespace bolt::api::core::limits
