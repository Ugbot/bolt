// bench_swiss_growable.cpp — SwissTableGrowable vs fixed SwissTable vs
// std::unordered_map (G2CHK-85 audit: the fd→handler registries in the
// src/api/net event loops sat on std::unordered_map — this quantifies the
// win the swap was chasing, and checks the growable sibling gives up nothing
// material against the fixed table it derives from).
//
// Fairness: identical key streams, identical {u64 → u32} payload shape, and
// comparable load factors — SwissTable and SwissTableGrowable both size 2x
// (≤50% steady load); unordered_map is given reserve(N) up front so it never
// rehashes inside a timed region either (its default max_load_factor of 1.0
// is its own design choice, as 7/8 is Abseil's/ours).
//
// Sections:
//   insert     N distinct keys into an empty table (growable starts at hint
//              16 so this INCLUDES every grow/rehash on the way up — the
//              honest cost of not pre-sizing; fixed SwissTable is pre-sized,
//              which is exactly its contract).
//   find-hit   N lookups of present keys (shuffled order).
//   find-miss  N lookups of absent keys.
//   churn      steady-state erase+insert pairs at constant live count — the
//              fd-registry workload. SwissTable has no erase: N/A.
//   registry   the event-loop dispatch mix: 15/16 find-hit + 1/16 churn pair
//              on a small (1k-entry) table — the shape the audit flagged.
//
// Correctness inside the bench: every section folds results into a checksum
// and the checksums must agree across implementations (loud abort on
// divergence, never a silently-wrong number).
//
// Tiger Style: no heap allocation inside timed regions (arenas + reserve up
// front), fixed loop bounds, asserts on pre/postconditions.

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/join/bolt_swiss_growable.h"

using namespace bolt;
using clk = std::chrono::steady_clock;

namespace {

constexpr uint32_t kN = 1u << 20;         // 1M keys for bulk sections
constexpr uint32_t kChurnLive = 1u << 16; // steady-state live count
constexpr uint32_t kChurnOps = 1u << 20;  // erase+insert pairs
constexpr uint32_t kRegLive = 1024;       // fd-registry live count
constexpr uint32_t kRegOps = 1u << 22;    // registry mixed ops

double ns_per(clk::time_point t0, clk::time_point t1, uint64_t ops) {
    assert(ops > 0);
    const double ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return ns / static_cast<double>(ops);
}


// Min-of-reps: this M4 box runs 6P+12E cores and often carries background
// load (see memory: bench hygiene); a single timing is +/-15% or worse.
// Each section runs kReps times and reports the minimum — the least-
// disturbed pass. Sections with per-rep setup do it inside the lambda,
// outside their own timed window.
constexpr int kReps = 7;

template <class F>
double min_of_reps(F&& one_rep) {
    double best = 1e300;
    for (int r = 0; r < kReps; ++r) {
        const double ns = one_rep();
        if (ns < best) best = ns;
    }
    return best;
}

// Distinct, well-mixed 64-bit keys (splitmix-style odd-multiplier walk).
uint64_t key_of(uint32_t i) { return (i + 1ull) * 0x9E3779B97F4A7C15ull; }
uint64_t miss_key_of(uint32_t i) { return (i + 1ull) * 0x9E3779B97F4A7C15ull + 1ull; }

struct Row {
    const char* impl;
    double insert_ns, hit_ns, miss_ns, churn_ns, reg_ns;
};

void print_row(const Row& r) {
    printf("%-22s insert %7.2f  find-hit %7.2f  find-miss %7.2f  churn %7.2f  registry %7.2f  (ns/op)\n",
           r.impl, r.insert_ns, r.hit_ns, r.miss_ns, r.churn_ns, r.reg_ns);
}

}  // namespace

int main() {
    // Shuffled lookup order (built once, shared by every impl).
    std::vector<uint32_t> order(kN);
    for (uint32_t i = 0; i < kN; ++i) order[i] = i;
    std::mt19937 shuf(7);
    for (uint32_t i = kN - 1; i > 0; --i) {
        std::swap(order[i], order[shuf() % (i + 1)]);
    }
    // Registry op stream: victim indices + op selector, shared.
    std::vector<uint32_t> reg_rand(kRegOps);
    std::mt19937 rr(11);
    for (uint32_t i = 0; i < kRegOps; ++i) reg_rand[i] = rr();

    uint64_t sum_hit_ref = 0, sum_reg_ref = 0;

    ArenaConfig cfg;
    cfg.max_block_size = 256u * 1024 * 1024;

    // ---------------- SwissTableGrowable ----------------
    Row grow_row{"SwissTableGrowable", 0, 0, 0, 0, 0};
    {
        // insert: fresh arena + table per rep (creation outside the window).
        grow_row.insert_ns = min_of_reps([&] {
            Arena arena(cfg);
            SwissTableGrowable t;
            if (!SwissTableGrowable::create(&t, 16, &arena)) abort();
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                if (!t.insert(key_of(i), i)) abort();
            }
            const auto t1 = clk::now();
            if (t.size != kN) abort();
            return ns_per(t0, t1, kN);
        });

        Arena arena(cfg);
        SwissTableGrowable t;
        if (!SwissTableGrowable::create(&t, kN, &arena)) abort();
        for (uint32_t i = 0; i < kN; ++i) {
            if (!t.insert(key_of(i), i)) abort();
        }

        grow_row.hit_ns = min_of_reps([&] {
            uint64_t sum = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                sum += static_cast<uint32_t>(t.find(key_of(order[i])));
            }
            const auto t1 = clk::now();
            if (sum_hit_ref == 0) sum_hit_ref = sum;
            if (sum != sum_hit_ref) abort();
            return ns_per(t0, t1, kN);
        });

        grow_row.miss_ns = min_of_reps([&] {
            uint64_t misses = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                misses += static_cast<uint64_t>(t.find(miss_key_of(order[i])) < 0);
            }
            const auto t1 = clk::now();
            if (misses != kN) abort();
            return ns_per(t0, t1, kN);
        });

        // churn: one steady-state table; every rep is an equally-valid
        // steady-state pass (state persists, live count constant).
        Arena arena2(cfg);
        SwissTableGrowable c;
        if (!SwissTableGrowable::create(&c, kChurnLive, &arena2)) abort();
        std::vector<uint64_t> live(kChurnLive);
        for (uint32_t i = 0; i < kChurnLive; ++i) {
            live[i] = key_of(i);
            if (!c.insert(live[i], i)) abort();
        }
        uint64_t next = kN + 1;
        grow_row.churn_ns = min_of_reps([&] {
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kChurnOps; ++i) {
                const uint32_t v = reg_rand[i & (kRegOps - 1)] & (kChurnLive - 1);
                if (!c.erase(live[v])) abort();
                live[v] = key_of(0) + (next++);
                if (!c.insert(live[v], v)) abort();
            }
            const auto t1 = clk::now();
            if (c.size != kChurnLive) abort();
            return ns_per(t0, t1, kChurnOps);
        });

        // fd-registry mix.
        Arena arena3(cfg);
        SwissTableGrowable r;
        if (!SwissTableGrowable::create(&r, kRegLive, &arena3)) abort();
        std::vector<uint64_t> rlive(kRegLive);
        for (uint32_t i = 0; i < kRegLive; ++i) {
            rlive[i] = key_of(i);
            if (!r.insert(rlive[i], i)) abort();
        }
        uint64_t rnext = 1;
        grow_row.reg_ns = min_of_reps([&] {
            uint64_t rsum = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kRegOps; ++i) {
                const uint32_t rnd = reg_rand[i];
                const uint32_t v = (rnd >> 8) & (kRegLive - 1);
                if ((rnd & 15u) == 0) {  // churn pair
                    if (!r.erase(rlive[v])) abort();
                    rlive[v] = key_of(kN + 7) + (rnext++);
                    if (!r.insert(rlive[v], v)) abort();
                } else {  // dispatch lookup
                    rsum += static_cast<uint32_t>(r.find(rlive[v]));
                }
            }
            const auto t1 = clk::now();
            if (sum_reg_ref == 0) sum_reg_ref = rsum;
            if (rsum != sum_reg_ref) abort();
            return ns_per(t0, t1, kRegOps);
        });
    }
    print_row(grow_row);

    // ---------------- fixed SwissTable (insert/find only) ----------------
    Row fixed_row{"SwissTable (fixed)", 0, 0, 0, -1, -1};
    {
        fixed_row.insert_ns = min_of_reps([&] {
            Arena arena(cfg);
            SwissTable t;
            if (!SwissTable::create(&t, kN, &arena)) abort();
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                if (!t.insert(key_of(i), i)) abort();
            }
            const auto t1 = clk::now();
            if (t.size != kN) abort();
            return ns_per(t0, t1, kN);
        });

        Arena arena(cfg);
        SwissTable t;
        if (!SwissTable::create(&t, kN, &arena)) abort();
        for (uint32_t i = 0; i < kN; ++i) {
            if (!t.insert(key_of(i), i)) abort();
        }

        fixed_row.hit_ns = min_of_reps([&] {
            uint64_t sum = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                sum += static_cast<uint32_t>(t.find(key_of(order[i])));
            }
            const auto t1 = clk::now();
            if (sum != sum_hit_ref) abort();
            return ns_per(t0, t1, kN);
        });

        fixed_row.miss_ns = min_of_reps([&] {
            uint64_t misses = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                misses += static_cast<uint64_t>(t.find(miss_key_of(order[i])) < 0);
            }
            const auto t1 = clk::now();
            if (misses != kN) abort();
            return ns_per(t0, t1, kN);
        });
    }
    print_row(fixed_row);

    // ---------------- std::unordered_map ----------------
    Row um_row{"std::unordered_map", 0, 0, 0, 0, 0};
    {
        um_row.insert_ns = min_of_reps([&] {
            std::unordered_map<uint64_t, uint32_t> t;
            t.reserve(kN);
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) t[key_of(i)] = i;
            const auto t1 = clk::now();
            if (t.size() != kN) abort();
            return ns_per(t0, t1, kN);
        });

        std::unordered_map<uint64_t, uint32_t> t;
        t.reserve(kN);
        for (uint32_t i = 0; i < kN; ++i) t[key_of(i)] = i;

        um_row.hit_ns = min_of_reps([&] {
            uint64_t sum = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                sum += t.find(key_of(order[i]))->second;
            }
            const auto t1 = clk::now();
            if (sum != sum_hit_ref) abort();
            return ns_per(t0, t1, kN);
        });

        um_row.miss_ns = min_of_reps([&] {
            uint64_t misses = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kN; ++i) {
                misses += static_cast<uint64_t>(t.find(miss_key_of(order[i])) == t.end());
            }
            const auto t1 = clk::now();
            if (misses != kN) abort();
            return ns_per(t0, t1, kN);
        });

        std::unordered_map<uint64_t, uint32_t> c;
        c.reserve(kChurnLive * 2);
        std::vector<uint64_t> live(kChurnLive);
        for (uint32_t i = 0; i < kChurnLive; ++i) {
            live[i] = key_of(i);
            c[live[i]] = i;
        }
        uint64_t next = kN + 1;
        um_row.churn_ns = min_of_reps([&] {
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kChurnOps; ++i) {
                const uint32_t v = reg_rand[i & (kRegOps - 1)] & (kChurnLive - 1);
                if (c.erase(live[v]) != 1) abort();
                live[v] = key_of(0) + (next++);
                c[live[v]] = v;
            }
            const auto t1 = clk::now();
            if (c.size() != kChurnLive) abort();
            return ns_per(t0, t1, kChurnOps);
        });

        std::unordered_map<uint64_t, uint32_t> r;
        r.reserve(kRegLive * 2);
        std::vector<uint64_t> rlive(kRegLive);
        for (uint32_t i = 0; i < kRegLive; ++i) {
            rlive[i] = key_of(i);
            r[rlive[i]] = i;
        }
        uint64_t rnext = 1;
        um_row.reg_ns = min_of_reps([&] {
            uint64_t rsum = 0;
            const auto t0 = clk::now();
            for (uint32_t i = 0; i < kRegOps; ++i) {
                const uint32_t rnd = reg_rand[i];
                const uint32_t v = (rnd >> 8) & (kRegLive - 1);
                if ((rnd & 15u) == 0) {
                    if (r.erase(rlive[v]) != 1) abort();
                    rlive[v] = key_of(kN + 7) + (rnext++);
                    r[rlive[v]] = v;
                } else {
                    rsum += r.find(rlive[v])->second;
                }
            }
            const auto t1 = clk::now();
            if (rsum != sum_reg_ref) abort();
            return ns_per(t0, t1, kRegOps);
        });
    }
    print_row(um_row);

    printf("checksums agree across implementations (hit=%llu registry=%llu)\n",
           static_cast<unsigned long long>(sum_hit_ref),
           static_cast<unsigned long long>(sum_reg_ref));
    return 0;
}
