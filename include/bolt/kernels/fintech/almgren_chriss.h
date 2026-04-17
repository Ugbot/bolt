// bolt/kernels/fintech/almgren_chriss.h — Phase 5.S.13.
//
// Almgren-Chriss optimal-execution schedule (Almgren & Chriss 2000).
// Produces a per-row slice allocation of a unit inventory `Q = 1` to be
// liquidated over a normalised horizon `T = 1` using the optimal minimum-
// variance solution under linear temporary impact `eta` and risk aversion
// `gamma` given instantaneous volatility `sigma`:
//
//     kappa = sqrt(gamma / (2 * eta * sigma^2))
//     x(t)  = Q * sinh(kappa * (T - t)) / sinh(kappa * T)
//     slice[k] = x(t_k) - x(t_{k+1}),   t_k = k * T / n_slices
//
// Degenerate fallback: if `2 * eta * sigma^2 < 1e-15` or `sinh(kappa*T) < 1e-15`,
// we fall back to the uniform schedule `slice[k] = Q / n_slices`. Matches
// chukonu exactly.
//
// Per-row shape: the kernel emits `slice[row_count % n_slices]` at every
// row. `row_count` persists across `execute_almgren_chriss` calls so a
// multi-batch stream walks the schedule cyclically with no seams.
//
// Input: NONE. AlmgrenChriss is a pure-schedule emitter — no input column
// is read. The `in` pointer array is accepted for signature parity with
// other Phase 5 kernels but is unused; this mirrors chukonu's shape
// exactly (it only reads `batch->num_rows()`).
//
// Branchless audit: inner loop is a single array lookup + modulo + store.
// Zero data-dependent branches. The modulo is a cheap integer operation
// (compiler emits imul/sub for small powers-of-two, idiv otherwise — in
// either case no branching). Schedule precomputation at make_* time uses
// sinh/sqrt but runs ONCE per graph compile, not per row.
//
// Source: chukonu/fintech/kernels.h :: CreateAlmgrenChrissProcessor (line 2408).
//
// Capacity: maximum supported n_slices. 4096 covers any practical trade
// granularity (1 slice per minute over a 68-hour horizon etc). Fits in
// 32 KiB which is comfortably L1-resident.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_port.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace bolt {
namespace fintech {

inline constexpr uint32_t kAlmgrenChrissMaxSlices = 4096u;
inline constexpr double   kAlmgrenChrissEps       = 1e-15;

struct AlmgrenChrissOpDesc {
    int32_t  out_col;
    uint32_t n_slices;
    double   sigma;
    double   gamma;
    double   eta;
};

struct AlmgrenChrissState {
    AlmgrenChrissOpDesc desc;
    double              schedule[kAlmgrenChrissMaxSlices];
    uint64_t            row_count;   // cumulative rows emitted, wraps via %
};

// Execute: emit `schedule[(row_count + i) % n_slices]` per row.
// No input column is read. `in` is accepted for signature parity and may
// even be nullptr — this is the ONLY Phase 5 kernel where that's legal.
inline void execute_almgren_chriss(AlmgrenChrissState* state,
                                   const double* BOLT_RESTRICT* /*in*/,
                                   int64_t n,
                                   double* BOLT_RESTRICT out) noexcept {
    assert(state != nullptr);
    assert(out != nullptr || n == 0);
    assert(n >= 0);
    assert(state->desc.n_slices > 0u);
    assert(state->desc.n_slices <= kAlmgrenChrissMaxSlices);

    const uint32_t ns      = state->desc.n_slices;
    uint64_t       counter = state->row_count;
    const double*  sched   = state->schedule;

    // Fully branchless per-row: load + modulo + store.
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t idx = static_cast<uint32_t>(counter % ns);
        out[i]             = sched[idx];
        ++counter;
    }
    state->row_count = counter;
}

// Precompute the schedule once. Caller picks n_slices <= kAlmgrenChrissMaxSlices.
inline AlmgrenChrissState* make_almgren_chriss_state(::bolt::Arena* arena,
                                                     int32_t  out_col,
                                                     uint32_t n_slices,
                                                     double   sigma,
                                                     double   gamma,
                                                     double   eta) noexcept {
    assert(arena != nullptr);
    assert(n_slices > 0u);
    assert(n_slices <= kAlmgrenChrissMaxSlices);
    AlmgrenChrissState* s = arena->allocate_array<AlmgrenChrissState>(1);
    if (s == nullptr) return nullptr;
    s->desc.out_col  = out_col;
    s->desc.n_slices = n_slices;
    s->desc.sigma    = sigma;
    s->desc.gamma    = gamma;
    s->desc.eta      = eta;
    s->row_count     = 0ull;

    const double Q      = 1.0;
    const double T      = 1.0;
    const double denom  = 2.0 * eta * sigma * sigma;
    const double kappa  = (denom > kAlmgrenChrissEps)
                              ? std::sqrt(gamma / denom)
                              : 0.0;
    const double sinhkT = std::sinh(kappa * T);

    // Degenerate case: uniform schedule (matches chukonu line 2429-2434).
    if (std::abs(sinhkT) < kAlmgrenChrissEps) {
        const double uniform = Q / static_cast<double>(n_slices);
        for (uint32_t k = 0; k < n_slices; ++k) s->schedule[k] = uniform;
        // Zero-fill the tail beyond n_slices for cleanliness.
        for (uint32_t k = n_slices; k < kAlmgrenChrissMaxSlices; ++k) {
            s->schedule[k] = 0.0;
        }
        return s;
    }

    const double dt = T / static_cast<double>(n_slices);
    for (uint32_t k = 0; k < n_slices; ++k) {
        const double t_k   = static_cast<double>(k) * dt;
        const double t_kp1 = static_cast<double>(k + 1u) * dt;
        const double x_k   = Q * std::sinh(kappa * (T - t_k  )) / sinhkT;
        const double x_kp1 = Q * std::sinh(kappa * (T - t_kp1)) / sinhkT;
        s->schedule[k]     = x_k - x_kp1;
    }
    for (uint32_t k = n_slices; k < kAlmgrenChrissMaxSlices; ++k) {
        s->schedule[k] = 0.0;
    }
    return s;
}

inline void almgren_chriss_reset(AlmgrenChrissState* state) noexcept {
    assert(state != nullptr);
    assert(state->desc.n_slices > 0u);
    state->row_count = 0ull;
}

}  // namespace fintech
}  // namespace bolt
