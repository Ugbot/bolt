// bolt_net_limits.h — fixed capacities of bolt::net, reported by the BOLT_LIMITS table (bolt_budget.h).
#pragma once

#include <cstdint>

namespace bolt::net {

// Per-call TLS send/receive deadline ceiling: tls_send_all_timeout / tls_recv_some_timeout refuse a
// larger value, so a longer request loops calls under its own deadline (G2CHK-1106).
constexpr std::uint32_t k_tls_io_timeout_max_ms = 60000u;

}  // namespace bolt::net
