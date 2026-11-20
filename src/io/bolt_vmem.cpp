// bolt_vmem.cpp — see bolt/io/bolt_vmem.h.

#include "bolt/io/bolt_vmem.h"

#include <cassert>
#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace bolt {
namespace io {

size_t vm_page_bytes() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<size_t>(si.dwPageSize);
#else
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<size_t>(p) : 4096u;
#endif
}

namespace {

size_t round_up(size_t v, size_t page) noexcept {
    assert(page != 0u && (page & (page - 1u)) == 0u);
    return (v + page - 1u) & ~(page - 1u);
}

}  // namespace

void* vm_reserve(size_t bytes) noexcept {
    if (bytes == 0u) return nullptr;
    const size_t n = round_up(bytes, vm_page_bytes());
    if (n < bytes) return nullptr;
#if defined(_WIN32)
    return VirtualAlloc(nullptr, n, MEM_RESERVE, PAGE_NOACCESS);
#else
    void* p = mmap(nullptr, n, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

bool vm_commit(void* base, size_t offset, size_t bytes) noexcept {
    if (base == nullptr) return false;
    if (bytes == 0u) return true;
    const size_t page = vm_page_bytes();
    if ((offset % page) != 0u || (bytes % page) != 0u) return false;
    void* at = static_cast<uint8_t*>(base) + offset;
#if defined(_WIN32)
    return VirtualAlloc(at, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return mprotect(at, bytes, PROT_READ | PROT_WRITE) == 0;
#endif
}

void vm_release(void* base, size_t bytes) noexcept {
    if (base == nullptr) return;
#if defined(_WIN32)
    (void)bytes;
    (void)VirtualFree(base, 0, MEM_RELEASE);
#else
    (void)munmap(base, round_up(bytes, vm_page_bytes()));
#endif
}

}  // namespace io
}  // namespace bolt
