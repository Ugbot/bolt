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

#include "ycpp/ycpp_arena.h"        // ycpp::Allocator concept
#include "ycpp/ycpp_awareness.h"
#include "ycpp/ycpp_byteview.h"
#include "ycpp/ycpp_delete_set.h"
#include "ycpp/ycpp_doc.h"          // Doc<A> + encode_diff_v1<A>
#include "ycpp/ycpp_envelope.h"
#include "ycpp/ycpp_protocol.h"
#include "ycpp/ycpp_state_vector.h"
#include "ycpp/ycpp_status.h"
#include "ycpp/ycpp_struct_store.h"
#include "ycpp/ycpp_subdoc.h"
#include "ycpp/ycpp_undo.h"
#include "ycpp/ycpp_update.h"
#include "ycpp/ycpp_update_v2.h"
#include "ycpp/ycpp_writer.h"
#include "ycpp/ycpp_yarray.h"
#include "ycpp/ycpp_ymap.h"
#include "ycpp/ycpp_ytext.h"

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
// Full production type aliases — every ycpp template instantiated against
// the bolt-arena-backed allocator. Decoder + encoder bodies live in the
// ycpp headers, so they monomorphise + inline at every call site here.
// -----------------------------------------------------------------------
using DeleteSet     = ::ycpp::DeleteSet     <BoltArenaAllocator>;
using StructStore   = ::ycpp::StructStore   <BoltArenaAllocator>;
using DecodedUpdate = ::ycpp::DecodedUpdate <BoltArenaAllocator>;
using StateVector   = ::ycpp::StateVector   <BoltArenaAllocator>;
using YMap          = ::ycpp::YMap          <BoltArenaAllocator>;
using YArray        = ::ycpp::YArray        <BoltArenaAllocator>;
using YText         = ::ycpp::YText         <BoltArenaAllocator>;
using Doc           = ::ycpp::Doc           <BoltArenaAllocator>;
using Awareness     = ::ycpp::Awareness     <BoltArenaAllocator>;
using UndoManager   = ::ycpp::UndoManager   <BoltArenaAllocator>;
using SubDocRegistry = ::ycpp::SubDocRegistry<BoltArenaAllocator>;

// Re-export the envelope + protocol surface verbatim so consumers can
// build RPC layers without naming `ycpp::` directly.
using ::ycpp::Envelope;
using ::ycpp::MessageKind;
using ::ycpp::encode_envelope;
using ::ycpp::decode_envelope;
using ::ycpp::emit_sync_update;

// -----------------------------------------------------------------------
// Wire-format helpers — thin wrappers that pin the allocator parameter to
// BoltArenaAllocator so callers don't have to spell the template each time.
// -----------------------------------------------------------------------
[[nodiscard]] inline ::ycpp::Status decode_update_v1(::ycpp::ByteView bytes,
                                                     DecodedUpdate*   out) noexcept {
    assert(out != nullptr);
    return ::ycpp::decode_update_v1<BoltArenaAllocator>(bytes, out);
}

[[nodiscard]] inline ::ycpp::Status encode_diff_v1(const Doc&         src,
                                                   const StateVector* since,
                                                   ::ycpp::Writer*    out) noexcept {
    assert(out != nullptr);
    return ::ycpp::encode_diff_v1<BoltArenaAllocator>(src, since, out);
}

// One-shot Doc construction: bolt::Arena + client_id → Doc. The arena
// MUST outlive the Doc; gestaltd typically scopes the arena to the
// session / room lifetime.
[[nodiscard]] inline Doc make_doc(::bolt::Arena& arena,
                                  std::uint64_t  client_id) noexcept {
    return Doc{client_id, BoltArenaAllocator{&arena}};
}

// Pinned wrappers for the sync-protocol emitters: hide the template
// allocator parameters so call sites stay clean. The `scratch` arena
// is used only for the SV staging; it never touches the Docs' arenas.
[[nodiscard]] inline ::ycpp::Status emit_sync_step1(const Doc&     doc,
                                                     ::bolt::Arena& scratch,
                                                     std::uint64_t  request_id,
                                                     ::ycpp::Writer* out) noexcept {
    BoltArenaAllocator a{&scratch};
    return ::ycpp::emit_sync_step1<BoltArenaAllocator, BoltArenaAllocator>(
        doc, a, request_id, out);
}

[[nodiscard]] inline ::ycpp::Status emit_sync_step2(const Doc&         doc,
                                                     const StateVector* peer_sv,
                                                     std::uint64_t      request_id,
                                                     ::ycpp::Writer*    out) noexcept {
    return ::ycpp::emit_sync_step2<BoltArenaAllocator>(doc, peer_sv, request_id, out);
}

[[nodiscard]] inline ::ycpp::Status apply_sync_message(Doc&             dst,
                                                       const Envelope&  env,
                                                       StateVector*     peer_sv_out) noexcept {
    return ::ycpp::apply_sync_message<BoltArenaAllocator>(dst, env, peer_sv_out);
}

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
