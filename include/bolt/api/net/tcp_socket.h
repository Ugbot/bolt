/**
 * Bolt API TCP Socket - High-level TCP abstraction
 *
 * Features:
 * - RAII wrapper around raw sockets
 * - Non-blocking I/O
 * - Zero-copy where possible
 * - Integration with EventLoop
 */

#pragma once

#include <cstdint>
#include <string>
#include "bolt/api/net/sys_compat.h"  // sockaddr_in, ssize_t (winsock on Windows)

namespace bolt::api {
namespace net {

/**
 * TCP Socket abstraction
 */
class TcpSocket {
public:
    /**
     * Create a TCP socket from an existing file descriptor
     * Takes ownership of the fd.
     */
    explicit TcpSocket(int fd);

    /**
     * Create a new TCP socket
     */
    TcpSocket();

    /**
     * Destructor closes the socket
     */
    ~TcpSocket();

    // Non-copyable, movable
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;
    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;

    /**
     * Get the file descriptor
     */
    int fd() const { return fd_; }

    /**
     * Check if socket is valid
     */
    bool is_valid() const { return fd_ >= 0; }

    /**
     * Close the socket
     */
    void close();

    /**
     * Set socket to non-blocking mode
     */
    int set_nonblocking();

    /**
     * Disable Nagle's algorithm (set TCP_NODELAY)
     */
    int set_nodelay();

    /**
     * Set SO_REUSEADDR
     */
    int set_reuseaddr();

    /**
     * Set SO_REUSEPORT (Linux only)
     */
    int set_reuseport();

    /**
     * Set SO_KEEPALIVE
     */
    int set_keepalive(bool enable = true);

    /**
     * Set receive buffer size
     */
    int set_recv_buffer_size(int size);

    /**
     * Set send buffer size
     */
    int set_send_buffer_size(int size);

    /**
     * Connect to a remote address
     * @param host Hostname or IP address
     * @param port Port number
     * @return 0 on success, -1 on error (check errno)
     */
    int connect(const std::string& host, uint16_t port);

    /**
     * Bind to local address
     * @param host Local IP address (or "0.0.0.0" for any)
     * @param port Local port
     * @return 0 on success, -1 on error
     */
    int bind(const std::string& host, uint16_t port);

    /**
     * Listen for connections
     * @param backlog Maximum pending connections
     * @return 0 on success, -1 on error
     */
    int listen(int backlog = 1024);

    /**
     * Accept a new connection
     * @param client_addr Will be filled with client address info
     * @return New TcpSocket for the connection, or invalid socket on error
     */
    TcpSocket accept(struct sockaddr_in* client_addr = nullptr);

    /**
     * Send data
     * @param data Pointer to data
     * @param len Length of data
     * @param flags Send flags (default 0)
     * @return Number of bytes sent, or -1 on error
     */
    ssize_t send(const void* data, size_t len, int flags = 0);

    /**
     * Receive data
     * @param buffer Buffer to receive into
     * @param len Maximum bytes to receive
     * @param flags Receive flags (default 0)
     * @return Number of bytes received, 0 on EOF, -1 on error
     */
    ssize_t recv(void* buffer, size_t len, int flags = 0);

    /**
     * Get local address
     */
    bool get_local_address(std::string& ip, uint16_t& port) const;

    /**
     * Get remote address
     */
    bool get_remote_address(std::string& ip, uint16_t& port) const;

    /**
     * Release ownership of the file descriptor
     * Returns the fd and sets internal fd to -1
     */
    int release();

private:
    int fd_;
};

} // namespace net
} // namespace bolt::api
