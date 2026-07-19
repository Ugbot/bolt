// bolt_row_view.h — Zero-copy row pointer + pin token.
//
// RULES: No exceptions. No RTTI. POD. Caller-stack allocated.
//
// A RowView is a fixed-capacity POD struct that carries column pointers
// directly into some underlying storage (memtable zone, pinned block-
// cache slot, Arrow chunked array, etc.) plus an opaque pin token that
// the owner uses to implement release semantics.
//
// Lifetime discipline:
//   1. Caller declares RowView on the stack (or from an arena).
//   2. Producer (e.g. `mt_get_view`) fills `col_ptrs[]`, `col_sizes[]`,
//      `ncols`, `key`, `ts_ns`, and the pin token.
//   3. Producer increments its own refcount (ZonePin for memtable,
//      BlockCache pin for SST) before returning. The RowView's
//      `pin_kind` + `pin_slot` + `pin_seq` let `release_row_view` find
//      the right counter to decrement.
//   4. Consumer reads fields directly — zero copy.
//   5. Consumer calls the release function (provided by the producer
//      lib) before letting the RowView go out of scope. If not
//      released, the owner will spin in `wait_drained` / eviction will
//      skip the slot — memory safety preserved, but progress stalls.
//
// Size: with `kMaxBatchColumns = 256` this is ~2.6 KiB. Within Bolt's
// norm for stack POD — `BoltBatch` is ~5 KiB on stack per bolt_column.h.
// Callers with fewer columns still only pay for the columns they fill;
// the rest of `col_ptrs` is irrelevant.
//
// ABA guard. Consumers of a RowView should not outlive the producer's
// slot generation. `pin_seq` captures the producer's slot seq at pin
// time; release must verify seq still matches before decrementing. If
// the slot was recycled (pin_seq changed) the reader has invalid data
// anyway, and the release is a no-op — fail-safe.

#pragma once

#include "bolt/bolt_column.h"   // kMaxBatchColumns
#include "bolt/bolt_port.h"

#include <cstdint>

namespace bolt {

// Pin kind discriminator. 0 = no pin (empty view, safe to drop).
enum class RowViewPinKind : uint8_t {
    kNone             = 0,
    kMemtableZone     = 1,
    kBlockCacheSlot   = 2,
    // Reserved for future owners (PDX cluster cache, etc.).
};

struct alignas(64) RowView {
    // Row metadata — populated by the producer on hit.
    int64_t     key;
    int64_t     ts_ns;

    // Column pointers + sizes, indexed 0..ncols-1. Positions beyond
    // `ncols` are undefined (do not read). Producers must match the
    // schema's column order.
    // G2FEAT-47: sized to kMaxFixedColumns (256), not the raised in-memory
    // kMaxBatchColumns — RowView is the point-get / KV row-materialisation
    // shape (schema-narrow), so it keeps the compact historical cap.
    const void* col_ptrs[kMaxFixedColumns];
    uint16_t    col_sizes[kMaxFixedColumns];
    uint32_t    ncols;

    // Opaque pin state. Consumers MUST NOT interpret these fields
    // directly — pass the RowView to the producer's release function
    // (e.g. `mdb_release_view`) which knows which pin-decrementor to
    // call based on `pin_kind`.
    RowViewPinKind pin_kind;
    uint8_t        _pad_pin[3];
    uint32_t       pin_slot;    // slot index (zone slot or block-cache slot)
    uint64_t       pin_seq;     // producer-side ABA guard
    void*          pin_ctx;     // primary back-pointer (Db* / SstReader*)
    void*          pin_ctx2;    // secondary — e.g. pinned BoltBatch* for
                                // block-cache release. Wave 18c.2.
};

// Stack-allocable. Bolt's convention: size assertions keep the shape
// stable for ABI-minded callers.
static_assert(sizeof(RowView) <=
                  (sizeof(int64_t) * 2 +
                   sizeof(void*) * kMaxFixedColumns +
                   sizeof(uint16_t) * kMaxFixedColumns +
                   sizeof(uint32_t) * 2 +
                   sizeof(uint64_t) +
                   sizeof(void*) +
                   128 /* alignment + pad slack */),
              "RowView layout drifted");

BOLT_FORCE_INLINE void row_view_init(RowView* v) noexcept {
    v->key       = 0;
    v->ts_ns     = 0;
    v->ncols     = 0u;
    v->pin_kind  = RowViewPinKind::kNone;
    v->pin_slot  = 0u;
    v->pin_seq   = 0u;
    v->pin_ctx   = nullptr;
    v->pin_ctx2  = nullptr;
}

// Read one column cell. Bounds-checked in debug builds; the ncols field
// must match the schema the producer knew about at fill time.
template <typename T>
BOLT_FORCE_INLINE T row_view_col(const RowView* v, uint32_t col) noexcept {
    assert(v != nullptr);
    assert(col < v->ncols);
    assert(v->col_sizes[col] == sizeof(T));
    T out;
    memcpy(&out, v->col_ptrs[col], sizeof(T));
    return out;
}

}  // namespace bolt
