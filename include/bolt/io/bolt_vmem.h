// bolt/io/bolt_vmem.h — reserved address space committed on demand
// (G2CHK-1140). A structure that must grow without moving (readers hold
// pointers into it with no lock) reserves its ceiling once, as address space
// only, and commits pages as it fills: no copy on growth, no realloc, and RSS
// follows use. Windows: VirtualAlloc MEM_RESERVE / MEM_COMMIT. POSIX: mmap
// PROT_NONE / mprotect.

#pragma once

#include <cstddef>

namespace bolt {
namespace io {

// Granularity of vm_commit offsets and lengths (the OS page size).
size_t vm_page_bytes() noexcept;

// Reserve `bytes` of address space (rounded up to a page). No memory is
// committed and no page is readable. Null on failure.
void* vm_reserve(size_t bytes) noexcept;

// Make [base + offset, base + offset + bytes) readable and writable, zero
// filled on first touch. `offset` and `bytes` are page multiples inside the
// reservation. Committing an already committed range is a no-op. False when
// the OS refuses (out of memory or commit charge).
bool vm_commit(void* base, size_t offset, size_t bytes) noexcept;

// Release a whole reservation (`bytes` as passed to vm_reserve).
void vm_release(void* base, size_t bytes) noexcept;

}  // namespace io
}  // namespace bolt
