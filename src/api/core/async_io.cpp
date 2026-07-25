/**
 * Async I/O - Factory and common implementation
 */

#include "bolt/api/core/async_io.h"
#include <iostream>
#include <mutex>

namespace bolt::api {
namespace core {

std::unique_ptr<async_io> async_io::create(const async_io_config& config) {
    io_backend backend = config.backend;
    
    // Auto-detect best backend for platform
    if (backend == io_backend::auto_detect) {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
        backend = io_backend::kqueue;
#elif defined(__linux__)
        // Prefer io_uring; the io_uring backend probes usable() at construction
        // and falls back to epoll at runtime when the kernel (<5.11) or a seccomp
        // policy (Docker default) rejects it — see the io_uring case below.
        backend = io_backend::io_uring;
#elif defined(_WIN32)
        backend = io_backend::iocp;
#else
        std::cerr << "No async I/O backend available for this platform" << std::endl;
        return nullptr;
#endif
    }
    
    // Create backend-specific implementation
    switch (backend) {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
        case io_backend::kqueue:
            return std::make_unique<kqueue_io>(config);
#endif
            
#ifdef __linux__
        case io_backend::epoll:
            return std::make_unique<epoll_io>(config);

        case io_backend::io_uring: {
            // Try the real io_uring backend; fall back to epoll if the kernel
            // (or container seccomp) won't let us. NEVER abort — Docker default
            // seccomp blocks io_uring_setup/enter (ENOSYS/EPERM) and old kernels
            // miss IORING_FEAT_EXT_ARG; both are expected and handled here.
            auto u = std::make_unique<io_uring_io>(config);
            if (u->usable()) {
                return u;
            }
            // Setup failed → epoll. Log once so test logs make the fallback
            // visible (no per-instance spam — every dispatcher would print it).
            static std::once_flag warned;
            std::call_once(warned, []() {
                std::cerr << "async_io: io_uring unavailable "
                             "(kernel <5.11 or seccomp blocked) — using epoll"
                          << std::endl;
            });
            return std::make_unique<epoll_io>(config);
        }
#endif
            
#ifdef _WIN32
        case io_backend::iocp:
            return std::make_unique<iocp_io>(config);
#endif
            
        default:
            std::cerr << "Unsupported async I/O backend" << std::endl;
            return nullptr;
    }
}

} // namespace core
} // namespace bolt::api

