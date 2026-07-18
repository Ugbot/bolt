/**
 * @file sys_compat.h
 * @brief Cross-platform socket/syscall compatibility shim.
 *
 * The Bolt API engine was hard-forked from a POSIX-first codebase that targets
 * (unistd.h / fcntl.h / sys/socket.h and the raw close/read/write/fcntl calls).
 * On Windows those names either do not exist or mean the wrong thing (e.g.
 * ::close on a SOCKET is a no-op / error; sockets are not fds). This header
 * provides a thin, header-only mapping so the forked files compile and run on
 * both Windows (winsock2) and POSIX without sprinkling #ifdefs everywhere.
 *
 * Usage: include this where raw POSIX socket calls happen and call
 *   bolt::api::net::sys::close_socket(fd)
 *   bolt::api::net::sys::recv_bytes(fd, buf, len)
 *   bolt::api::net::sys::send_bytes(fd, buf, len)
 *   bolt::api::net::sys::set_nonblocking(fd)
 *   bolt::api::net::sys::startup()   // WSAStartup once; no-op on POSIX
 *
 * On Windows the forked bodies that called raw close()/read()/write() are
 * rewritten to use the bolt::api::net::sys helpers below.
 */

#pragma once

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>   // socklen_t, inet_pton/ntop, getaddrinfo
#include <mswsock.h>
#include <windows.h>
#include <io.h>
#include <cstdint>
#include <mutex>

// <windows.h> defines a pile of ALL-CAPS macros that collide with identifiers
// used as enumerators / member names in the forked engine (e.g. IOEvent::ERROR,
// HTTP1Method::DELETE, parser states). Undefine the offenders so the engine's
// scoped enums and methods parse correctly. We do not use the Win32 versions.
#ifdef DELETE
#undef DELETE
#endif
#ifdef ERROR
#undef ERROR
#endif
#ifdef OPTIONS
#undef OPTIONS
#endif
#ifdef CONNECT
#undef CONNECT
#endif
#ifdef IGNORE
#undef IGNORE
#endif
#ifdef NO_ERROR
#undef NO_ERROR
#endif
#ifdef DOMAIN
#undef DOMAIN
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// ssize_t is POSIX; MSVC provides SSIZE_T in BaseTsd.h. Define a portable alias.
#include <BaseTsd.h>
#ifndef BOLTAPI_HAVE_SSIZE_T
#define BOLTAPI_HAVE_SSIZE_T 1
using ssize_t = SSIZE_T;
#endif

// STDOUT_FILENO etc. used by async-signal-safe writes in the server.
#ifndef STDIN_FILENO
#define STDIN_FILENO  0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

// ---------------------------------------------------------------------------
// POSIX time / byte-order helpers that MSVC lacks. Provided as inline shims so
// the forked bodies (gmtime_r / be64toh / timegm / strptime …) compile.
// ---------------------------------------------------------------------------
#include <ctime>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

inline struct tm* boltapi_gmtime_r(const time_t* t, struct tm* out) {
    return (::gmtime_s(out, t) == 0) ? out : nullptr;
}
inline struct tm* boltapi_localtime_r(const time_t* t, struct tm* out) {
    return (::localtime_s(out, t) == 0) ? out : nullptr;
}
#ifndef gmtime_r
#define gmtime_r(t, out)    boltapi_gmtime_r((t), (out))
#endif
#ifndef localtime_r
#define localtime_r(t, out) boltapi_localtime_r((t), (out))
#endif

#ifndef timegm
#define timegm(tm) (::_mkgmtime(tm))
#endif

#ifndef be64toh
#define be64toh(x) (_byteswap_uint64(x))
#endif
#ifndef htobe64
#define htobe64(x) (_byteswap_uint64(x))
#endif
#ifndef be32toh
#define be32toh(x) (_byteswap_ulong(x))
#endif
#ifndef htobe32
#define htobe32(x) (_byteswap_ulong(x))
#endif
#ifndef be16toh
#define be16toh(x) (_byteswap_ushort(x))
#endif
#ifndef htobe16
#define htobe16(x) (_byteswap_ushort(x))
#endif

// Minimal strptime: supports only the HTTP/RFC1123 date format used by the
// engine ("%a, %d %b %Y %H:%M:%S GMT"). Returns end pointer or nullptr.
inline char* boltapi_strptime_http(const char* s, const char* fmt, struct tm* tm) {
    (void)fmt;  // we only implement the single format the engine passes
    static const char* months[] = {"Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"};
    char mon[4] = {0};
    int day = 0, year = 0, hour = 0, minute = 0, second = 0;
    // Skip leading weekday + ", "
    const char* comma = std::strchr(s, ',');
    const char* p = comma ? comma + 1 : s;
    while (*p == ' ') ++p;
    if (std::sscanf(p, "%d %3s %d %d:%d:%d", &day, mon, &year, &hour, &minute, &second) != 6) {
        return nullptr;
    }
    int mon_idx = -1;
    for (int i = 0; i < 12; ++i) {
        if (std::strncmp(mon, months[i], 3) == 0) { mon_idx = i; break; }
    }
    if (mon_idx < 0) return nullptr;
    tm->tm_mday = day;
    tm->tm_mon  = mon_idx;
    tm->tm_year = year - 1900;
    tm->tm_hour = hour;
    tm->tm_min  = minute;
    tm->tm_sec  = second;
    return const_cast<char*>(s + std::strlen(s));
}
#ifndef strptime
#define strptime(s, fmt, tm) boltapi_strptime_http((s), (fmt), (tm))
#endif

namespace bolt::api {
namespace net {
namespace sys {

/// Initialize Winsock exactly once. Safe to call repeatedly.
inline void startup() noexcept {
    static std::once_flag once;
    std::call_once(once, []() noexcept {
        WSADATA wsa;
        ::WSAStartup(MAKEWORD(2, 2), &wsa);
    });
}

/// Close a socket handle.
inline int close_socket(SOCKET fd) noexcept {
    return ::closesocket(fd);
}

/// Receive bytes from a socket (POSIX read() equivalent).
inline ssize_t recv_bytes(SOCKET fd, void* buf, size_t len) noexcept {
    return ::recv(fd, static_cast<char*>(buf), static_cast<int>(len), 0);
}

/// Send bytes to a socket (POSIX write() equivalent).
inline ssize_t send_bytes(SOCKET fd, const void* buf, size_t len) noexcept {
    return ::send(fd, static_cast<const char*>(buf), static_cast<int>(len), 0);
}

/// Put a socket into non-blocking mode (POSIX fcntl(O_NONBLOCK) equivalent).
inline int set_nonblocking(SOCKET fd) noexcept {
    u_long mode = 1;
    return ::ioctlsocket(fd, FIONBIO, &mode);
}

}  // namespace sys
}  // namespace net
}  // namespace bolt::api

// NOTE: we deliberately do NOT #define close/read/write here. Those names
// collide with C++ member functions used widely in the engine (e.g.
// RingBuffer::read, socket.close()). Call sites are instead rewritten to use
// the bolt::api::net::sys helpers above (close_socket / recv_bytes /
// send_bytes / set_nonblocking).

#else  // !_WIN32

#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <cstddef>

namespace bolt::api {
namespace net {
namespace sys {

inline void startup() noexcept {}

inline int close_socket(int fd) noexcept { return ::close(fd); }

inline ssize_t recv_bytes(int fd, void* buf, size_t len) noexcept {
    return ::read(fd, buf, len);
}

inline ssize_t send_bytes(int fd, const void* buf, size_t len) noexcept {
    return ::write(fd, buf, len);
}

inline int set_nonblocking(int fd) noexcept {
    int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

}  // namespace sys
}  // namespace net
}  // namespace bolt::api

#endif  // _WIN32
