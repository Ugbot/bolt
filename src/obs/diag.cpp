// diag.cpp — see diag.h.
#include "bolt/obs/diag.h"

#if !defined(_WIN32)

#include <arpa/inet.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace bolt::diag {

namespace {

constexpr int kAcceptPollMs = 100;
constexpr int kReadTimeoutMs = 1000;
constexpr std::size_t kRequestBytes = 2048;

struct State {
    std::atomic<bool> running{false};
    std::thread thread;
    int listen_fd = -1;
    Handler handler = nullptr;
    void* ctx = nullptr;
};
State g;

void write_all(int fd, const char* p, std::size_t n) noexcept {
    while (n > 0) {
        const ssize_t k = ::send(fd, p, n, 0);
        if (k <= 0) return;
        p += k;
        n -= static_cast<std::size_t>(k);
    }
}

void serve(int fd) {
    char req[kRequestBytes];
    pollfd pfd{fd, POLLIN, 0};
    if (::poll(&pfd, 1, kReadTimeoutMs) <= 0) return;
    const ssize_t n = ::recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 4) return;
    req[n] = '\0';
    if (std::strncmp(req, "GET ", 4) != 0) return;
    char* path = req + 4;
    char* end = std::strpbrk(path, " ?\r\n");
    if (end != nullptr) *end = '\0';
    std::string body;
    const char* type = "application/json";
    const int status = g.handler(g.ctx, path, &body, &type);
    char head[192];
    const int h = std::snprintf(head, sizeof(head), "HTTP/1.0 %d X\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                                status, type, body.size());
    write_all(fd, head, static_cast<std::size_t>(h));
    write_all(fd, body.data(), body.size());
}

void main_loop() {
    while (g.running.load(std::memory_order_acquire)) {
        pollfd pfd{g.listen_fd, POLLIN, 0};
        if (::poll(&pfd, 1, kAcceptPollMs) <= 0) continue;
        const int fd = ::accept(g.listen_fd, nullptr, nullptr);
        if (fd < 0) continue;
        serve(fd);
        ::close(fd);
    }
}

}  // namespace

bool start(std::uint16_t port, bool all_interfaces, Handler handler, void* ctx, std::uint16_t* bound) noexcept {
    assert(handler != nullptr);
    if (g.running.load(std::memory_order_acquire)) return false;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    const int one = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(all_interfaces ? INADDR_ANY : INADDR_LOOPBACK);
    a.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || ::listen(fd, 16) != 0) { ::close(fd); return false; }
    socklen_t len = sizeof(a);
    if (bound != nullptr && ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0) *bound = ntohs(a.sin_port);
    g.listen_fd = fd;
    g.handler = handler;
    g.ctx = ctx;
    g.running.store(true, std::memory_order_release);
    g.thread = std::thread(main_loop);
    return true;
}

void stop() noexcept {
    if (!g.running.exchange(false, std::memory_order_acq_rel)) return;
    g.thread.join();
    ::close(g.listen_fd);
    g.listen_fd = -1;
}

}  // namespace bolt::diag

#else

namespace bolt::diag {
bool start(std::uint16_t, bool, Handler, void*, std::uint16_t*) noexcept { return false; }
void stop() noexcept {}
}  // namespace bolt::diag

#endif
