// Private HTTP client limits, included inside bolt::net anonymous namespace.
// uint32_t is supplied by the including translation unit.
#pragma once

// Kind C: maximum per-exchange timeout budget.
constexpr uint32_t kMaxTimeoutMs = 60000;

// Kind C: bounded wire-framing overhead byte budget.
constexpr uint32_t kMaxWireOverhead = 64u * 1024u;

// Kind B: initial response read-buffer allocation shape.
constexpr uint32_t kInitialReadCapacity = 8192;

// Kind C: bounded interrupted/would-block socket retry budget.
constexpr uint32_t kMaxSocketRetries = 64;

namespace http_client_limits {
// Kind B: fixed request-head scratch storage; function-local alias retains its scope.
constexpr uint32_t kRequestHeadCap = 16u * 1024u;
}  // namespace http_client_limits
