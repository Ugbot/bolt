/**
 * IOCP-based async I/O implementation (Windows)
 * 
 * High-performance async I/O using I/O Completion Ports.
 */

#include "bolt/api/core/async_io.h"

#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "mswsock.lib")

namespace bolt::api {
namespace core {

/**
 * OVERLAPPED structure with operation data
 */
struct iocp_op {
    OVERLAPPED overlapped;
    io_op operation;
    SOCKET sock;
    io_callback callback;
    void* user_data;
    
    // For read/write
    WSABUF wsabuf;
    char* buffer{nullptr};
    DWORD bytes_transferred{0};
    DWORD flags{0};
    
    // For accept
    SOCKET accept_socket{INVALID_SOCKET};
    char accept_buffer[2 * (sizeof(sockaddr_in) + 16)];
    
    // For connect
    struct sockaddr_storage addr;
    int addrlen{0};

    // For recvfrom (UDP): WSARecvFrom fills `from_addr`/`from_len` (which must
    // outlive the async op — they live in this heap-allocated op). On
    // completion we copy them back into the caller's out-params below.
    struct sockaddr_storage from_addr;
    int                     from_len{0};
    struct sockaddr*        user_src{nullptr};   // caller out: peer address
    socklen_t*              user_srclen{nullptr}; // caller in/out: capacity/len

    iocp_op() { reset(); }

    // Re-initialize for reuse from the pool. Clears the OVERLAPPED (required —
    // the kernel reads it), the recvfrom out-param scratch, and the per-op
    // fields a fresh submit sets. `callback` is reset by the pool on free.
    void reset() noexcept {
        ZeroMemory(&overlapped, sizeof(overlapped));
        operation = io_op::read;
        sock = INVALID_SOCKET;
        user_data = nullptr;
        wsabuf.buf = nullptr;
        wsabuf.len = 0;
        buffer = nullptr;
        bytes_transferred = 0;
        flags = 0;
        accept_socket = INVALID_SOCKET;
        ZeroMemory(&addr, sizeof(addr));
        addrlen = 0;
        ZeroMemory(&from_addr, sizeof(from_addr));
        from_len = sizeof(from_addr);
        user_src = nullptr;
        user_srclen = nullptr;
    }
};

// Completion keys to distinguish wake/stop events from real I/O
static constexpr ULONG_PTR WAKE_KEY = 0xFADE0001;
static constexpr ULONG_PTR STOP_KEY = 0xFADE0002;

/**
 * IOCP implementation
 */
struct iocp_io::impl {
    HANDLE iocp_handle{INVALID_HANDLE_VALUE};
    async_io_config config;

    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Wake mechanism
    async_io::wake_callback wake_cb;
    std::atomic<bool> wake_pending{false};
    
    // AcceptEx function pointer (loaded dynamically)
    LPFN_ACCEPTEX AcceptEx{nullptr};
    LPFN_CONNECTEX ConnectEx{nullptr};
    
    // Statistics
    std::atomic<uint64_t> stat_accepts{0};
    std::atomic<uint64_t> stat_reads{0};
    std::atomic<uint64_t> stat_writes{0};
    std::atomic<uint64_t> stat_connects{0};
    std::atomic<uint64_t> stat_closes{0};
    std::atomic<uint64_t> stat_polls{0};
    std::atomic<uint64_t> stat_events{0};
    std::atomic<uint64_t> stat_errors{0};
    
    impl(const async_io_config& cfg) : config(cfg) {
        // Initialize Winsock
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        
        // Create IOCP
        iocp_handle = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
        if (iocp_handle == NULL) {
            // Handle error
        }
        
        // Load AcceptEx and ConnectEx
        load_extension_functions();
    }
    
    ~impl() {
        if (iocp_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(iocp_handle);
        }
        // Drain the op free list (these are recycled, not in-flight, ops).
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (iocp_op* op : pool_free) delete op;
            pool_free.clear();
        }
        WSACleanup();
    }
    
    void load_extension_functions() {
        // Create temporary socket to get extension functions
        SOCKET temp_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (temp_socket == INVALID_SOCKET) return;
        
        // Get AcceptEx
        GUID acceptex_guid = WSAID_ACCEPTEX;
        DWORD bytes;
        WSAIoctl(temp_socket, SIO_GET_EXTENSION_FUNCTION_POINTER,
                &acceptex_guid, sizeof(acceptex_guid),
                &AcceptEx, sizeof(AcceptEx),
                &bytes, NULL, NULL);
        
        // Get ConnectEx
        GUID connectex_guid = WSAID_CONNECTEX;
        WSAIoctl(temp_socket, SIO_GET_EXTENSION_FUNCTION_POINTER,
                &connectex_guid, sizeof(connectex_guid),
                &ConnectEx, sizeof(ConnectEx),
                &bytes, NULL, NULL);
        
        closesocket(temp_socket);
    }
    
    // Sockets already associated with this IOCP. CreateIoCompletionPort is a
    // syscall and only needs to run ONCE per handle; re-associating on every
    // submit (e.g. per-datagram recvfrom on a long-lived UDP socket) is pure
    // overhead and fails with ERROR_INVALID_PARAMETER anyway. Cache the set so
    // the hot path skips the syscall after the first association.
    std::mutex          assoc_mutex;
    std::unordered_map<SOCKET, char> associated;

    int associate_socket(SOCKET sock) {
        {
            std::lock_guard<std::mutex> lock(assoc_mutex);
            if (associated.find(sock) != associated.end()) {
                return 0;  // already associated — skip the CreateIoCompletionPort syscall
            }
        }
        HANDLE h = CreateIoCompletionPort((HANDLE)sock, iocp_handle, (ULONG_PTR)sock, 0);
        const bool ok = (h == iocp_handle);
        if (ok) {
            std::lock_guard<std::mutex> lock(assoc_mutex);
            associated.emplace(sock, char{1});
        }
        return ok ? 0 : -1;
    }

    void forget_socket(SOCKET sock) {
        std::lock_guard<std::mutex> lock(assoc_mutex);
        associated.erase(sock);
    }

    // ---- iocp_op free-list pool ------------------------------------------
    // new/delete on the per-datagram hot path is expensive (CLAUDE.md). Recycle
    // completed iocp_op objects through a lock-guarded free list. The lock is
    // only contended between the I/O thread (which frees on completion and may
    // alloc on re-arm) and submit threads; it is far cheaper than malloc/free.
    std::mutex             pool_mutex;
    std::vector<iocp_op*>  pool_free;
    static constexpr std::size_t kPoolCap = 256;  // bounded retained free slots

    iocp_op* alloc_op() {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (!pool_free.empty()) {
                iocp_op* op = pool_free.back();
                pool_free.pop_back();
                op->reset();
                return op;
            }
        }
        return new (std::nothrow) iocp_op();
    }

    void free_op(iocp_op* op) {
        if (op == nullptr) return;
        op->callback = nullptr;  // release any captured state promptly
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (pool_free.size() < kPoolCap) {
                pool_free.push_back(op);
                return;
            }
        }
        delete op;
    }
};

iocp_io::iocp_io(const async_io_config& config)
    : impl_(std::make_unique<impl>(config)) {
}

iocp_io::~iocp_io() {
    stop();
}

int iocp_io::accept_async(
    int listen_fd,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET listen_socket = (SOCKET)listen_fd;
    
    // Associate with IOCP
    impl_->associate_socket(listen_socket);
    
    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::accept;
    op->sock = listen_socket;
    op->callback = std::move(callback);
    op->user_data = user_data;

    // Create accept socket
    op->accept_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (op->accept_socket == INVALID_SOCKET) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    // Start accept operation
    DWORD bytes_received = 0;
    BOOL result = impl_->AcceptEx(
        listen_socket,
        op->accept_socket,
        op->accept_buffer,
        0,  // No initial receive
        sizeof(sockaddr_in) + 16,
        sizeof(sockaddr_in) + 16,
        &bytes_received,
        &op->overlapped
    );
    
    if (!result && WSAGetLastError() != ERROR_IO_PENDING) {
        closesocket(op->accept_socket);
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_accepts.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::read_async(
    int fd,
    void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET sock = (SOCKET)fd;
    impl_->associate_socket(sock);
    
    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::read;
    op->sock = sock;
    op->buffer = static_cast<char*>(buffer);
    op->callback = std::move(callback);
    op->user_data = user_data;

    op->wsabuf.buf = op->buffer;
    op->wsabuf.len = static_cast<ULONG>(size);

    // Start receive operation
    DWORD flags = 0;
    int result = WSARecv(sock, &op->wsabuf, 1, &op->bytes_transferred, &flags, &op->overlapped, NULL);

    if (result != 0 && WSAGetLastError() != WSA_IO_PENDING) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::write_async(
    int fd,
    const void* buffer,
    size_t size,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET sock = (SOCKET)fd;
    impl_->associate_socket(sock);
    
    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::write;
    op->sock = sock;
    op->buffer = const_cast<char*>(static_cast<const char*>(buffer));
    op->callback = std::move(callback);
    op->user_data = user_data;

    op->wsabuf.buf = op->buffer;
    op->wsabuf.len = static_cast<ULONG>(size);

    // Start send operation
    int result = WSASend(sock, &op->wsabuf, 1, &op->bytes_transferred, 0, &op->overlapped, NULL);

    if (result != 0 && WSAGetLastError() != WSA_IO_PENDING) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    
    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::connect_async(
    int fd,
    const struct sockaddr* addr,
    socklen_t addrlen,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET sock = (SOCKET)fd;
    impl_->associate_socket(sock);
    
    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::connect;
    op->sock = sock;
    op->callback = std::move(callback);
    op->user_data = user_data;
    memcpy(&op->addr, addr, addrlen);
    op->addrlen = addrlen;
    
    // Bind to any address (required for ConnectEx)
    struct sockaddr_in local_addr;
    ZeroMemory(&local_addr, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = INADDR_ANY;
    local_addr.sin_port = 0;
    bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr));
    
    // Start connect operation
    BOOL result = impl_->ConnectEx(
        sock,
        (struct sockaddr*)&op->addr,
        op->addrlen,
        NULL,  // No send buffer
        0,
        NULL,
        &op->overlapped
    );
    
    if (!result && WSAGetLastError() != ERROR_IO_PENDING) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_connects.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::recvfrom_async(
    int fd,
    void* buffer,
    size_t size,
    struct sockaddr* src,
    socklen_t* srclen,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET sock = (SOCKET)fd;
    impl_->associate_socket(sock);

    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::recvfrom;
    op->sock = sock;
    op->buffer = static_cast<char*>(buffer);
    op->callback = std::move(callback);
    op->user_data = user_data;
    op->user_src = src;
    op->user_srclen = srclen;

    op->wsabuf.buf = op->buffer;
    op->wsabuf.len = static_cast<ULONG>(size);

    // WSARecvFrom: the OVERLAPPED, the WSABUF, AND the from-addr/from-len
    // out-params must all stay valid until the completion is dequeued — they
    // live inside `op`, which we only delete in poll() after the completion.
    DWORD flags = 0;
    int result = WSARecvFrom(
        sock,
        &op->wsabuf, 1,
        &op->bytes_transferred,
        &flags,
        reinterpret_cast<sockaddr*>(&op->from_addr),
        &op->from_len,
        &op->overlapped,
        NULL);

    if (result != 0 && WSAGetLastError() != WSA_IO_PENDING) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_reads.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::sendto_async(
    int fd,
    const void* buffer,
    size_t size,
    const struct sockaddr* dst,
    socklen_t dstlen,
    io_callback callback,
    void* user_data
) noexcept {
    SOCKET sock = (SOCKET)fd;
    impl_->associate_socket(sock);

    auto* op = impl_->alloc_op();
    if (op == nullptr) { impl_->stat_errors.fetch_add(1, std::memory_order_relaxed); return -1; }
    op->operation = io_op::sendto;
    op->sock = sock;
    op->buffer = const_cast<char*>(static_cast<const char*>(buffer));
    op->callback = std::move(callback);
    op->user_data = user_data;

    op->wsabuf.buf = op->buffer;
    op->wsabuf.len = static_cast<ULONG>(size);

    // Destination address must outlive the op; copy it into the op storage.
    if (dst != nullptr && dstlen > 0 &&
        dstlen <= static_cast<socklen_t>(sizeof(op->addr))) {
        memcpy(&op->addr, dst, dstlen);
        op->addrlen = dstlen;
    }

    int result = WSASendTo(
        sock,
        &op->wsabuf, 1,
        &op->bytes_transferred,
        0,
        reinterpret_cast<sockaddr*>(&op->addr),
        op->addrlen,
        &op->overlapped,
        NULL);

    if (result != 0 && WSAGetLastError() != WSA_IO_PENDING) {
        impl_->free_op(op);
        impl_->stat_errors.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    impl_->stat_writes.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

int iocp_io::close_async(int fd) noexcept {
    impl_->stat_closes.fetch_add(1, std::memory_order_relaxed);
    // Drop the association cache entry: the SOCKET value may be recycled by the
    // OS for a future socket, which must be (re-)associated on its first op.
    impl_->forget_socket((SOCKET)fd);
    return closesocket((SOCKET)fd);
}

int iocp_io::poll(uint32_t timeout_us) noexcept {
    if (impl_->iocp_handle == INVALID_HANDLE_VALUE) return -1;
    
    impl_->stat_polls.fetch_add(1, std::memory_order_relaxed);
    
    DWORD timeout_ms = timeout_us / 1000;
    
    // Get completion status
    DWORD bytes_transferred;
    ULONG_PTR completion_key;
    LPOVERLAPPED overlapped;
    
    BOOL result = GetQueuedCompletionStatus(
        impl_->iocp_handle,
        &bytes_transferred,
        &completion_key,
        &overlapped,
        timeout_ms
    );
    
    if (!result && overlapped == NULL) {
        // Timeout or error
        return 0;
    }

    // Check for wake/stop signals (no OVERLAPPED, special completion key)
    if (overlapped == NULL || completion_key == WAKE_KEY || completion_key == STOP_KEY) {
        if (completion_key == WAKE_KEY) {
            impl_->wake_pending.store(false, std::memory_order_release);
            if (impl_->wake_cb) {
                impl_->wake_cb();
            }
        }
        // STOP_KEY: just return, the run() loop checks stop_requested
        return (completion_key == WAKE_KEY) ? 1 : 0;
    }

    // Get operation from OVERLAPPED. We recycle it back to the pool AFTER the
    // callback returns (the callback may itself submit a new op, which draws a
    // DIFFERENT op from the pool — safe, since this one is still in use until it
    // returns).
    iocp_op* op = CONTAINING_RECORD(overlapped, iocp_op, overlapped);

    impl_->stat_events.fetch_add(1, std::memory_order_relaxed);
    
    // Create event
    io_event event;
    event.operation = op->operation;
    event.fd = (int)op->sock;
    event.user_data = op->user_data;
    event.flags = 0;
    
    if (!result) {
        // Operation failed
        event.result = -1;
    } else {
        switch (op->operation) {
            case io_op::accept:
                event.result = (ssize_t)op->accept_socket;
                break;
                
            case io_op::read:
            case io_op::write:
            case io_op::sendto:
                event.result = bytes_transferred;
                break;

            case io_op::recvfrom:
                event.result = bytes_transferred;
                // Copy the peer address WSARecvFrom captured back into the
                // caller's out-params (truncating to their capacity).
                if (op->user_src != nullptr && op->user_srclen != nullptr) {
                    socklen_t cap = *op->user_srclen;
                    socklen_t got = static_cast<socklen_t>(op->from_len);
                    socklen_t copy = (got < cap) ? got : cap;
                    if (copy > 0) {
                        memcpy(op->user_src, &op->from_addr, copy);
                    }
                    *op->user_srclen = got;
                } else if (op->user_srclen != nullptr) {
                    *op->user_srclen = static_cast<socklen_t>(op->from_len);
                }
                break;

            case io_op::connect:
                event.result = 0;  // Success
                break;
                
            default:
                event.result = 0;
                break;
        }
    }
    
    // Invoke callback (inline on the I/O thread), then recycle the op.
    if (op->callback) {
        op->callback(event);
    }
    impl_->free_op(op);

    return 1;
}

void iocp_io::run() noexcept {
    if (impl_->running.exchange(true)) {
        return;  // Already running
    }
    
    impl_->stop_requested.store(false);
    
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        poll(impl_->config.poll_timeout_us);
    }
    
    impl_->running.store(false);
}

void iocp_io::stop() noexcept {
    impl_->stop_requested.store(true, std::memory_order_release);

    // Post stop signal to wake up IOCP
    if (impl_->iocp_handle != INVALID_HANDLE_VALUE) {
        PostQueuedCompletionStatus(impl_->iocp_handle, 0, STOP_KEY, NULL);
    }
}

bool iocp_io::is_running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}

async_io::stats iocp_io::get_stats() const noexcept {
    stats s;
    s.accepts = impl_->stat_accepts.load(std::memory_order_relaxed);
    s.reads = impl_->stat_reads.load(std::memory_order_relaxed);
    s.writes = impl_->stat_writes.load(std::memory_order_relaxed);
    s.connects = impl_->stat_connects.load(std::memory_order_relaxed);
    s.closes = impl_->stat_closes.load(std::memory_order_relaxed);
    s.polls = impl_->stat_polls.load(std::memory_order_relaxed);
    s.events = impl_->stat_events.load(std::memory_order_relaxed);
    s.errors = impl_->stat_errors.load(std::memory_order_relaxed);
    return s;
}

void iocp_io::wake() noexcept {
    // Thread-safe wake using PostQueuedCompletionStatus with WAKE_KEY.
    // Only post if not already pending (avoid redundant wakes).
    bool expected = false;
    if (impl_->wake_pending.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        if (impl_->iocp_handle != INVALID_HANDLE_VALUE) {
            PostQueuedCompletionStatus(impl_->iocp_handle, 0, WAKE_KEY, NULL);
        }
    }
}

void iocp_io::set_wake_callback(wake_callback callback) noexcept {
    impl_->wake_cb = std::move(callback);
}

} // namespace core
} // namespace bolt::api

#endif // _WIN32









