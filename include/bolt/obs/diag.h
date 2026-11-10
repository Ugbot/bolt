// bolt/obs/diag.h — a minimal HTTP/1.0 responder on its own thread, for processes
// with no gateway (workers) and for gateways still starting up. GET only, one
// request per connection, no keep-alive. Off unless a port is given.
#pragma once

#include <cstdint>
#include <string>

namespace bolt::diag {

// Fills body and content type; returns the status code. Runs on the diag thread.
using Handler = int (*)(void* ctx, const char* path, std::string* body, const char** content_type);

// Binds 127.0.0.1 unless all_interfaces. port 0 picks one; *bound receives it.
bool start(std::uint16_t port, bool all_interfaces, Handler handler, void* ctx, std::uint16_t* bound) noexcept;
void stop() noexcept;

}  // namespace bolt::diag
