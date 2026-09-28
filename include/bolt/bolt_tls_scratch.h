#pragma once
// bolt_tls_scratch.h — per-thread scratch that does NOT live in static TLS.
//
// glibc carves every thread's static TLS block (all `thread_local` storage in
// the process image) out of that thread's stack mapping, so a large
// `static thread_local T x` is stack every thread gives up — including threads
// that never touch it. Only an 8-byte pointer lives in static TLS here; the
// table is heap-allocated on first use by a thread and freed at thread exit.
//
// tls_scratch<Tag, T, N>() returns a value-initialised T[N] owned by the
// calling thread, or nullptr if that one allocation fails (callers fail their
// operation). Tag keeps two call sites with the same T and N apart.
// Same contract as chukonu::util::tls_scratch (G2ICE-165 / G2CHK-143).

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace bolt {

template <class Tag, class T, std::size_t N = 1>
T* tls_scratch() noexcept {
    static_assert(N > 0, "tls_scratch: empty table");
    static_assert(std::is_trivially_destructible_v<T>,
                  "tls_scratch: T is freed without running destructors");
    static_assert(alignof(T) <= alignof(std::max_align_t),
                  "tls_scratch: over-aligned T needs aligned_alloc");
    struct Holder {
        T* p = nullptr;
        ~Holder() { std::free(p); }
    };
    thread_local Holder h;
    if (h.p != nullptr) return h.p;
    void* mem = std::malloc(sizeof(T) * N);
    if (mem == nullptr) return nullptr;
    T* arr = static_cast<T*>(mem);
    for (std::size_t i = 0; i < N; ++i) ::new (static_cast<void*>(&arr[i])) T();
    h.p = arr;
    assert(h.p != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(h.p) % alignof(T) == 0u);
    return h.p;
}

// tls_grow<Tag, T>(n) returns a per-thread buffer of at least n T, growing
// x2 when a larger n arrives (contents are NOT preserved across growth: call
// it at a query boundary, then hand the span to the per-row loop). Freed at
// thread exit. nullptr when the allocation fails or n overflows size_t.
template <class Tag, class T>
T* tls_grow(std::size_t n) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "tls_grow: T is reused without constructors");
    static_assert(alignof(T) <= alignof(std::max_align_t),
                  "tls_grow: over-aligned T needs aligned_alloc");
    struct Holder {
        T*          p   = nullptr;
        std::size_t cap = 0;
        ~Holder() { std::free(p); }
    };
    thread_local Holder h;
    if (n == 0) n = 1;
    if (n <= h.cap) return h.p;
    if (n > SIZE_MAX / sizeof(T) / 2) return nullptr;
    std::size_t want = h.cap * 2;
    if (want < n) want = n;
    void* mem = std::malloc(want * sizeof(T));
    if (mem == nullptr) return nullptr;
    std::free(h.p);
    h.p = static_cast<T*>(mem);
    h.cap = want;
    assert(h.cap >= n);
    assert(h.p != nullptr);
    return h.p;
}

}  // namespace bolt
