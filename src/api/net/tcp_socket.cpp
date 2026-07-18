/**
 * Bolt API TCP Socket - Implementation
 */

#include "bolt/api/net/tcp_socket.h"
#include "bolt/api/net/event_loop.h"
#include "bolt/api/net/sys_compat.h"
#ifndef _WIN32
#include <netinet/tcp.h>
#endif
#include <cstring>
#include <errno.h>

namespace bolt::api {
namespace net {

TcpSocket::TcpSocket(int fd)
    : fd_(fd)
{
}

TcpSocket::TcpSocket()
    : fd_(-1)
{
    sys::startup();
    fd_ = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
}

TcpSocket::~TcpSocket() {
    close();
}

TcpSocket::TcpSocket(TcpSocket&& other) noexcept
    : fd_(other.fd_)
{
    other.fd_ = -1;
}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void TcpSocket::close() {
    if (fd_ >= 0) {
        sys::close_socket(fd_);
        fd_ = -1;
    }
}

int TcpSocket::set_nonblocking() {
    return EventLoop::set_nonblocking(fd_);
}

int TcpSocket::set_nodelay() {
    return EventLoop::set_tcp_nodelay(fd_);
}

int TcpSocket::set_reuseaddr() {
    return EventLoop::set_reuseaddr(fd_);
}

int TcpSocket::set_reuseport() {
    return EventLoop::set_reuseport(fd_);
}

int TcpSocket::set_keepalive(bool enable) {
    int val = enable ? 1 : 0;
    if (setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<const char*>(&val), sizeof(val)) < 0) {
        return -1;
    }
    return 0;
}

int TcpSocket::set_recv_buffer_size(int size) {
    if (setsockopt(fd_, SOL_SOCKET, SO_RCVBUF,
                   reinterpret_cast<const char*>(&size), sizeof(size)) < 0) {
        return -1;
    }
    return 0;
}

int TcpSocket::set_send_buffer_size(int size) {
    if (setsockopt(fd_, SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&size), sizeof(size)) < 0) {
        return -1;
    }
    return 0;
}

int TcpSocket::connect(const std::string& host, uint16_t port) {
    if (fd_ < 0) {
        errno = EBADF;
        return -1;
    }

    // Resolve hostname
    struct addrinfo hints, *result;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(host.c_str(), nullptr, &hints, &result);
    if (ret != 0) {
        errno = EINVAL;
        return -1;
    }

    // Use first result
    struct sockaddr_in addr;
    std::memcpy(&addr, result->ai_addr, sizeof(addr));
    addr.sin_port = htons(port);
    freeaddrinfo(result);

    // Connect
    if (::connect(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        if (errno != EINPROGRESS) {
            return -1;
        }
    }

    return 0;
}

int TcpSocket::bind(const std::string& host, uint16_t port) {
    if (fd_ < 0) {
        errno = EBADF;
        return -1;
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (host == "0.0.0.0" || host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            errno = EINVAL;
            return -1;
        }
    }

    if (::bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        return -1;
    }

    return 0;
}

int TcpSocket::listen(int backlog) {
    if (fd_ < 0) {
        errno = EBADF;
        return -1;
    }

    if (::listen(fd_, backlog) < 0) {
        return -1;
    }

    return 0;
}

TcpSocket TcpSocket::accept(struct sockaddr_in* client_addr) {
    if (fd_ < 0) {
        errno = EBADF;
        return TcpSocket(-1);
    }

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    int client_fd = static_cast<int>(::accept(fd_, (struct sockaddr*)&addr, &addr_len));

    if (client_addr) {
        *client_addr = addr;
    }

    return TcpSocket(client_fd);
}

ssize_t TcpSocket::send(const void* data, size_t len, int flags) {
    if (fd_ < 0) {
        errno = EBADF;
        return -1;
    }

    return ::send(fd_, reinterpret_cast<const char*>(data),
                  static_cast<int>(len), flags);
}

ssize_t TcpSocket::recv(void* buffer, size_t len, int flags) {
    if (fd_ < 0) {
        errno = EBADF;
        return -1;
    }

    return ::recv(fd_, reinterpret_cast<char*>(buffer),
                  static_cast<int>(len), flags);
}

bool TcpSocket::get_local_address(std::string& ip, uint16_t& port) const {
    if (fd_ < 0) {
        return false;
    }

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getsockname(fd_, (struct sockaddr*)&addr, &addr_len) < 0) {
        return false;
    }

    char ip_str[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str)) == nullptr) {
        return false;
    }

    ip = ip_str;
    port = ntohs(addr.sin_port);
    return true;
}

bool TcpSocket::get_remote_address(std::string& ip, uint16_t& port) const {
    if (fd_ < 0) {
        return false;
    }

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getpeername(fd_, (struct sockaddr*)&addr, &addr_len) < 0) {
        return false;
    }

    char ip_str[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str)) == nullptr) {
        return false;
    }

    ip = ip_str;
    port = ntohs(addr.sin_port);
    return true;
}

int TcpSocket::release() {
    int fd = fd_;
    fd_ = -1;
    return fd;
}

} // namespace net
} // namespace bolt::api
