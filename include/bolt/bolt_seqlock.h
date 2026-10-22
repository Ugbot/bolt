// bolt_seqlock.h — single-writer, many-reader atomic snapshot.
//
// RULES: No exceptions. No RTTI. Zero allocation.
//
// Motivation. When a single thread publishes multi-word state that
// multiple threads read concurrently — e.g. `TableRuntime::ssts[]` +
// `num_ssts` together, or a WAL segment-table snapshot — a mutex is
// overkill and blocks readers. A seqlock lets readers take a
// consistent snapshot by:
//   1. Read the sequence number.
//   2. If odd, writer is mid-write → retry.
//   3. Read the value.
//   4. Re-read the sequence number. If changed, retry.
//
// Writer:
//   1. fetch_add(1) on the sequence number → odd (writing).
//   2. Write the value.
//   3. fetch_add(1) again → even (stable).
//
// Two atomic fetch_add instructions plus release payload stores per write.
// Readers acquire atomic payload cells; sequence validation alone cannot make
// a concurrent plain memcpy legal in the C++ memory model.
//
// This is a template on the protected type `T`, which must be
// trivially copyable. No allocation or reader lock. Snapshot consistency does
// not pin pointees: callers retain their existing mapping/object lifetime pins.

#pragma once

#include "bolt/bolt_config.h"
#include "bolt/bolt_port.h"
#include "bolt/bolt_sequence.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace bolt {

template <typename T>
struct alignas(alignof(T) > bolt::config::kCacheIsolationBytes
    ? alignof(T) : bolt::config::kCacheIsolationBytes) Seqlock {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Seqlock<T> requires trivially copyable T");
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "Seqlock requires lock-free 64-bit atomic cells");
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
    static constexpr std::size_t kCellBytes = sizeof(uint64_t);
    static constexpr std::size_t kCells = (sizeof(T) + kCellBytes - 1u) / kCellBytes;
    static constexpr std::size_t kPayloadAlignment = alignof(T) > bolt::config::kCacheIsolationBytes
        ? alignof(T) : bolt::config::kCacheIsolationBytes;

    Sequence          seq;     // even = stable, odd = writing
    alignas(kPayloadAlignment) std::atomic<uint64_t> value[kCells];

    // Exclusive initialization, completed before publishing this object.
    BOLT_FORCE_INLINE void init(const T& initial) noexcept {
        assert(kCells != 0u);
        assert(kPayloadAlignment >= alignof(std::atomic<uint64_t>));
        store_payload(initial);
        seq.store_relaxed(0);
    }

    // Writer (single-writer contract). Two atomic increments per
    // write. Writer is responsible for ensuring no concurrent writer.
    BOLT_FORCE_INLINE void write(const T& v) noexcept {
        [[maybe_unused]] const uint64_t s0 = seq.fetch_add_acq_rel(1);   // → odd
        assert((s0 & 1u) == 0u);
        assert(kCells != 0u);
        store_payload(v);
        seq.fetch_add_acq_rel(1);   // → even, publishes the complete epoch
    }

    // At most 128 attempts. On success, out is one completed epoch. On false,
    // out is unspecified and must not be used as a validated snapshot.
    //
    // Proof (assuming no full uint64_t sequence wrap during one attempt):
    // first acquire observes even E, so E's cell stores happen-before loads;
    // cell coherence excludes older values. If a cell observes later writer W,
    // its release/acquire edge puts W's preceding odd sequence before the final
    // sequence load. Sequence coherence then excludes the old E: retry. Equal
    // even samples therefore accept only E, without racing on payload bytes.
    BOLT_FORCE_INLINE bool try_read(T* out) const noexcept {
        assert(out != nullptr);
        assert(kCells != 0u);
        for (int tries = 0; tries < 128; ++tries) {
            const uint64_t s1 = seq.load_acquire();
            if (s1 & 1u) { BOLT_PAUSE(); continue; }  // writer in progress
            auto* bytes = reinterpret_cast<unsigned char*>(out);
            for (std::size_t i = 0u; i < kCells; ++i) {
                const uint64_t cell = value[i].load(std::memory_order_acquire);
                const std::size_t offset = i * kCellBytes;
                const std::size_t count = sizeof(T) - offset < kCellBytes ? sizeof(T) - offset : kCellBytes;
                std::memcpy(bytes + offset, &cell, count);
            }
            const uint64_t s2 = seq.load_acquire();
            if (s1 == s2) return true;
            BOLT_PAUSE();
        }
        return false;
    }

    // Legacy spin-until-success wrapper. Requires writer progress and a stable
    // interval; neither bounded nor wait-free under a stalled/continuous writer.
    BOLT_FORCE_INLINE T read() const noexcept {
        T out;
        while (!try_read(&out)) { BOLT_PAUSE(); }
        return out;
    }

private:
    BOLT_FORCE_INLINE void store_payload(const T& input) noexcept {
        assert(kCells != 0u);
        assert(sizeof(T) <= kCells * kCellBytes);
        const auto* bytes = reinterpret_cast<const unsigned char*>(&input);
        for (std::size_t i = 0u; i < kCells; ++i) {
            const std::size_t offset = i * kCellBytes;
            const std::size_t count = sizeof(T) - offset < kCellBytes ? sizeof(T) - offset : kCellBytes;
            uint64_t cell = 0u; // final cell never reads beyond the caller's T
            std::memcpy(&cell, bytes + offset, count);
            value[i].store(cell, std::memory_order_release);
        }
    }
};

}  // namespace bolt
