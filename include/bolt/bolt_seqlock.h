// bolt_seqlock.h — Linux-kernel-style seqlock. Single-writer, many-
// reader, wait-free on the read side.
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
// Two atomic fetch_add instructions per write — no mutex, no wait.
//
// This is a template on the protected type `T`, which must be
// trivially copyable (we memcpy it in/out). For larger state use an
// SeqlockPtr variant that swaps an atomic pointer.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/bolt_sequence.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace bolt {

template <typename T>
struct alignas(64) Seqlock {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Seqlock<T> requires trivially copyable T");

    Sequence          seq;     // even = stable, odd = writing
    alignas(64) T     value;

    BOLT_FORCE_INLINE void init(const T& initial) noexcept {
        seq.store_relaxed(0);
        std::memcpy(&value, &initial, sizeof(T));
    }

    // Writer (single-writer contract). Two atomic increments per
    // write. Writer is responsible for ensuring no concurrent writer.
    BOLT_FORCE_INLINE void write(const T& v) noexcept {
        [[maybe_unused]] const uint64_t s0 = seq.fetch_add_acq_rel(1);   // → odd
        assert((s0 & 1u) == 0u);
        std::memcpy(&value, &v, sizeof(T));
        seq.fetch_add_acq_rel(1);                        // → even
    }

    // Reader — retries under concurrent write. On success writes into
    // `out` and returns true. Always eventually succeeds (lock-free).
    //
    // Memory ordering: the acquire fence between the payload memcpy and
    // the s2 load ensures prior data reads are sequenced before the
    // second sequence load — otherwise MSVC/x64 was reordering per-word
    // memcpy loads across the atomic and slipping ~0.15% torn snapshots
    // past the s1==s2 check under contention. Mirrors Linux's
    // smp_rmb() between data read and re-read of the seq counter.
    BOLT_FORCE_INLINE bool try_read(T* out) const noexcept {
        assert(out != nullptr);
        for (int tries = 0; tries < 128; ++tries) {
            const uint64_t s1 = seq.load_acquire();
            if (s1 & 1u) { BOLT_PAUSE(); continue; }  // writer in progress
            std::memcpy(out, &value, sizeof(T));
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t s2 = seq.load_acquire();
            if (s1 == s2) return true;
            BOLT_PAUSE();
        }
        return false;
    }

    // Blocking read — spin-retries until success. Always succeeds.
    BOLT_FORCE_INLINE T read() const noexcept {
        T out;
        while (!try_read(&out)) { BOLT_PAUSE(); }
        return out;
    }
};

}  // namespace bolt
