// bolt/ybolt.h — Y CRDT runtime adapter atop bolt::Arena + ycpp.
//
// ybolt is bolt's production binding for the standalone, open-source
// `ycpp` Yjs CRDT runtime (https://github.com/Ugbot/ycpp). The split:
//
//   ycpp  — algorithm + wire format, zero external deps, MIT
//   ybolt — bolt-backed allocator + fionn-JSON helpers, lives inside bolt
//
// The adapter is intentionally thin. ycpp is policy-based on its
// `Allocator` concept; ybolt's `BoltArenaAllocator` is the policy that
// makes ycpp's Doc allocate through `bolt::Arena`. Every allocation site
// in ycpp is monomorphised + inlined by the compiler when instantiated
// against this allocator — no virtual call, no FFI hop.
//
// Layout invariant: ycpp::ByteView and bolt::ByteView are both
// `{ const uint8_t*, size_t }`. We document the seam and provide
// adapter helpers; we do NOT alias the types (so consumers can always
// `static_cast` rather than reinterpret).

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"

#include "ycpp/ycpp_arena.h"   // ycpp::Allocator concept
#include "ycpp/ycpp_byteview.h"
#include "ycpp/ycpp_delete_set.h"
#include "ycpp/ycpp_status.h"
#include "ycpp/ycpp_struct_store.h"
#include "ycpp/ycpp_update.h"  // DecodedUpdate (template; decoder TU in W3)
// NOTE: ycpp_doc.h is W3-in-progress; once tested it adds:
//   using Doc        = ::ycpp::Doc<BoltArenaAllocator>;
//   using StateVector = ::ycpp::StateVector<BoltArenaAllocator>;
//   using YMap        = ::ycpp::YMap<BoltArenaAllocator>;
// For now ybolt exposes only the W1+W2 surface — proven solid above.

namespace bolt::ybolt {

// -----------------------------------------------------------------------
// BoltArenaAllocator — satisfies ycpp::Allocator over bolt::Arena.
//
// One `bolt::Arena*` per ycpp::Doc. The Arena outlives the Doc; we never
// own the arena (the embedder does — typically a per-request, per-
// fragment, or per-Doc lifetime managed by gestaltd).
//
// `free()` is a no-op: bolt::Arena is bump-only. Per-object reclamation
// belongs to ycpp's typed pools, which sit a level above this allocator.
//
// The static_assert at the bottom locks in concept conformance — if a
// future ycpp version changes the Allocator contract, this file fails
// fast at compile time.
// -----------------------------------------------------------------------
class BoltArenaAllocator {
public:
    explicit BoltArenaAllocator(::bolt::Arena* a) noexcept : arena_(a) {
        assert(a != nullptr);
    }

    BoltArenaAllocator(const BoltArenaAllocator&)            = default;
    BoltArenaAllocator& operator=(const BoltArenaAllocator&) = default;
    BoltArenaAllocator(BoltArenaAllocator&&) noexcept            = default;
    BoltArenaAllocator& operator=(BoltArenaAllocator&&) noexcept = default;
    ~BoltArenaAllocator() noexcept = default;

    [[nodiscard]] void* alloc(std::size_t n, std::size_t align) noexcept {
        assert(arena_ != nullptr);
        assert(align > 0);
        assert((align & (align - 1)) == 0 && "align must be power of two");
        if (n == 0) return nullptr;
        return arena_->allocate(n, align);
    }

    void free(void* /*p*/, std::size_t /*n*/) noexcept {
        // bump-only — bolt::Arena reclaims everything at reset() or destroy.
    }

    [[nodiscard]] std::size_t bytes_in_use() const noexcept {
        assert(arena_ != nullptr);
        return arena_->total_allocated();
    }

    [[nodiscard]] ::bolt::Arena* arena() const noexcept { return arena_; }

private:
    ::bolt::Arena* arena_;
};

static_assert(::ycpp::Allocator<BoltArenaAllocator>,
              "BoltArenaAllocator must satisfy ycpp::Allocator");

// -----------------------------------------------------------------------
// Type aliases — production W2 surface backed by bolt::Arena.
// W3 (in flight) adds Doc, YMap, StateVector once their tests are green.
// `decode_update_v1` lands here once the impl moves to an .inl so non-
// default-allocator instantiations are visible at the call site.
// -----------------------------------------------------------------------
using DeleteSet    = ::ycpp::DeleteSet<BoltArenaAllocator>;
using StructStore  = ::ycpp::StructStore<BoltArenaAllocator>;
using DecodedUpdate = ::ycpp::DecodedUpdate<BoltArenaAllocator>;

// -----------------------------------------------------------------------
// ByteView adapters. Layout is identical, but we expose explicit casts
// rather than aliasing the types — that way consumers retain control.
// -----------------------------------------------------------------------
[[nodiscard]] inline ::ycpp::ByteView from_bolt(const uint8_t* p, std::size_t n) noexcept {
    return ::ycpp::ByteView{p, n};
}

[[nodiscard]] inline ::ycpp::ByteView from_cstr(const char* s) noexcept {
    if (s == nullptr) return ::ycpp::ByteView{};
    std::size_t n = 0;
    while (s[n] != '\0') ++n;
    return ::ycpp::ByteView{reinterpret_cast<const uint8_t*>(s), n};
}

} // namespace bolt::ybolt
