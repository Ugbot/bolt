#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt::api::net::limits {

// Kind B: compile-time allocation bounds for loop-owned storage. Adjusting
// these requires rebuilding the loop and its consumers together. No queue
// grows during poll/dispatch: excess readiness stays in the kernel, and a
// nested dispatch beyond the pin stack is explicitly refused.
inline constexpr uint32_t kMaxDispatchDepth = 16;
inline constexpr size_t kEventBatchCapacity = 256;

static_assert(kMaxDispatchDepth > 0);
static_assert(kEventBatchCapacity > 0 && kEventBatchCapacity <= INT32_MAX);

}  // namespace bolt::api::net::limits
