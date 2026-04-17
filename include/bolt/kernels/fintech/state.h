// bolt/kernels/fintech/state.h — Phase 2 fintech state substrate
//
// Three POD state types that every Tier 2 fintech kernel composes on:
//   RollingRing<T, kCap>   — fixed-capacity ring buffer + running sum.
//   WelfordAccumulator     — online (count, mean, m2) via Welford's algorithm.
//   EmaState               — single-value exponential running state.
//
// Tiger-Style rules enforced here:
//   - POD layout (trivial ctor/dtor, zero-init via = {} or init()).
//   - Fixed-capacity buffers via constexpr template params; no heap.
//   - noexcept everywhere; no exceptions, no RTTI, no virtuals.
//   - Arena-pinned at graph-compile time — state lifetime is the graph's
//     lifetime. These structs never allocate.
//   - Hard upper bound kCap <= kMaxRollingRingCap (Tiger Style).
//   - >=2 asserts per hot-path method; functions <= 70 lines.
//
// The state types are deliberately decoupled from the kernel execute_*
// functions: each execute_<name> (Phase 3+) embeds the relevant state
// inside its <Kernel>State POD and calls update() on it.

#pragma once

#include "bolt/bolt_port.h"
#include "bolt/kernels/bolt_numeric.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace bolt {
namespace fintech {

// Upper bound on any RollingRing capacity. Picked to keep the ring in L1/L2
// even at sizeof(double) — 65536 * 8 B = 512 KiB worst case. Kernels that
// need larger windows must specialise (not anticipated for Tier 2).
inline constexpr uint32_t kMaxRollingRingCap = 65536u;

// ============================================================================
// RollingRing<T, kCap> — fixed-capacity ring buffer with O(1) running sum.
// ============================================================================
// Maintains the most-recent `effective_size` samples (configured via init()).
// Running sum updated incrementally: on eviction, subtract the evicted
// element; on every push, add the new element. No full re-sum.
//
// Numerical note (floating point): the "subtract the evicted value" approach
// accumulates drift over very long streams. For production stats use a
// WelfordAccumulator instead of mean()/sum from this ring. The ring's
// sum-path is adequate for SMA-class kernels where drift is bounded by the
// window length and typical signal magnitude.
template <typename T, uint32_t kCap>
struct RollingRing {
    static_assert(kCap > 0, "RollingRing capacity must be > 0");
    static_assert(kCap <= kMaxRollingRingCap,
                  "RollingRing capacity exceeds kMaxRollingRingCap");

    using Accum = ::bolt::kernels::accum_t<T>;

    T         ring[kCap];
    uint32_t  pos;              // next write index in [0, effective_size)
    uint32_t  count;            // current fill in [0, effective_size]
    uint32_t  effective_size;   // configured window, <= kCap
    Accum     sum;              // running sum over the current fill

    // Zero every field. Sets the effective window size. `size` must satisfy
    // 0 < size <= kCap. Caller is expected to pin the ring in graph arena
    // once and call init() once at graph compile time.
    inline void init(uint32_t size) noexcept {
        assert(size > 0u);
        assert(size <= kCap);
        effective_size = size;
        pos   = 0u;
        count = 0u;
        sum   = Accum{0};
        for (uint32_t i = 0; i < kCap; ++i) ring[i] = T{0};
    }

    inline bool full()  const noexcept { return count == effective_size; }
    inline uint32_t size() const noexcept { return count; }
    inline uint32_t capacity() const noexcept { return effective_size; }

    // Append one sample. When the ring is full, the oldest sample at `pos`
    // is evicted and subtracted from `sum`; the new sample is written at
    // `pos` and added to `sum`. When the ring is not yet full, the new
    // sample is added at the end.
    inline void push(T v) noexcept {
        assert(effective_size > 0u);
        assert(pos < effective_size);
        if (count < effective_size) {
            ring[pos] = v;
            sum += static_cast<Accum>(v);
            pos = (pos + 1u) % effective_size;
            ++count;
        } else {
            const T evicted = ring[pos];
            sum -= static_cast<Accum>(evicted);
            ring[pos] = v;
            sum += static_cast<Accum>(v);
            pos = (pos + 1u) % effective_size;
        }
    }

    // Mean over the currently-retained samples. Requires count > 0.
    inline double mean() const noexcept {
        assert(count > 0u);
        assert(effective_size > 0u);
        return static_cast<double>(sum) / static_cast<double>(count);
    }

    // Most-recent sample offset by k (0 = most recent, 1 = previous, ...).
    // Requires k < count; used by Lag / Autocorr / BipowerVar.
    inline T back(uint32_t k) const noexcept {
        assert(effective_size > 0u);
        assert(k < count);
        // `pos` points to the next write slot, so the most-recent element
        // lives at (pos - 1) mod effective_size. Offset by k further back.
        const uint32_t idx = (pos + effective_size - 1u - k) % effective_size;
        return ring[idx];
    }
};

// ============================================================================
// WelfordAccumulator — single-variable running (count, mean, m2).
// ============================================================================
// Welford's online algorithm (Knuth TAOCP vol. 2, 4.2.2). Numerically stable
// vs. the naive E[x^2] - E[x]^2 form, which suffers catastrophic cancellation
// when variance << mean^2. Welford is our default for every variance-based
// fintech kernel (RollingStd, RollingZScore, Sharpe, Sortino, ...).
//
// variance_pop()    = m2 / count            (dividing by N; assumes full pop)
// variance_sample() = m2 / (count - 1)      (Bessel-corrected, unbiased sample)
//
// Default choice across Bolt fintech kernels is variance_sample() because
// tick data is treated as a sample drawn from an unknown underlying process;
// the 1/(N-1) form is the statistically conventional default for vol / Sharpe
// / Sortino calculations that Bolt's Tier-2 kernels produce.
struct WelfordAccumulator {
    uint64_t count;
    double   mean;
    double   m2;

    inline void init() noexcept {
        count = 0ull;
        mean  = 0.0;
        m2    = 0.0;
    }

    // Welford update:
    //   count <- count + 1
    //   delta  = x - mean
    //   mean  <- mean + delta / count
    //   delta2 = x - mean              (post-update mean)
    //   m2    <- m2 + delta * delta2
    inline void update(double x) noexcept {
        assert(!(x != x) || x == x);      // tolerate NaN propagation explicitly
        assert(count < 0xFFFFFFFFFFFFFFFFull);
        ++count;
        const double delta  = x - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    // Population variance. Requires count > 0.
    inline double variance_pop() const noexcept {
        assert(count > 0ull);
        assert(m2 >= 0.0 || m2 != m2);     // m2 may be NaN if inputs were NaN
        return m2 / static_cast<double>(count);
    }

    // Sample variance (Bessel-corrected). Requires count > 1.
    inline double variance_sample() const noexcept {
        assert(count > 1ull);
        assert(m2 >= 0.0 || m2 != m2);
        return m2 / static_cast<double>(count - 1ull);
    }

    inline double stddev_pop() const noexcept {
        assert(count > 0ull);
        assert(m2 >= 0.0 || m2 != m2);
        return std::sqrt(variance_pop());
    }

    inline double stddev_sample() const noexcept {
        assert(count > 1ull);
        assert(m2 >= 0.0 || m2 != m2);
        return std::sqrt(variance_sample());
    }
};

// ============================================================================
// EmaState — single-value exponential running state.
// ============================================================================
// Classic EMA with alpha = 2 / (period + 1). First sample seeds the state
// (no warmup ramp), subsequent samples follow
//     value <- alpha * x + (1 - alpha) * value
// Returns the new running value from update() for convenience so the caller
// doesn't have to re-read the struct.
//
// Caller pins EmaState in graph arena once, calls init(period) once, and
// then update(x) per sample across the graph's lifetime.
struct EmaState {
    double alpha;
    double value;
    bool   initialized;

    // Configure alpha from a period in samples. period must be >= 1.
    // period = 1 gives alpha = 1 (every sample fully replaces state),
    // which degenerates to the identity function.
    inline void init(int32_t period) noexcept {
        assert(period >= 1);
        assert(period <= (int32_t)1 << 30);
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        value = 0.0;
        initialized = false;
    }

    // Re-arm the state without changing alpha; next update() seeds again.
    inline void reset() noexcept {
        assert(alpha >= 0.0);
        assert(alpha <= 1.0);
        value = 0.0;
        initialized = false;
    }

    // Feed one sample and return the new running value.
    inline double update(double x) noexcept {
        assert(alpha >= 0.0);
        assert(alpha <= 1.0);
        if (!initialized) {
            value = x;
            initialized = true;
        } else {
            value = alpha * x + (1.0 - alpha) * value;
        }
        return value;
    }
};

}  // namespace fintech
}  // namespace bolt
