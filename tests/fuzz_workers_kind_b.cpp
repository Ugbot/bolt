// fuzz_workers_kind_b — seeded fuzzer for L8 (worker count and topology as
// kind B, G2CHK-339). A run is fully determined by its seed; a failure prints
// the seed and `--seed N` replays it.
//
//   limits    random BOLT_WORKERS-style overrides (primary + legacy alias,
//             junk, overflow, edges) through limits_resolve, against an
//             independent parser: valid -> kEnv, invalid -> kInvalid with the
//             variable named, never a silently clamped value.
//   scheduler random worker counts (incl. past the ceiling, reinit after a
//             refusal, pinning on/off) and random submit_range shapes: every
//             row runs exactly once on a tid below thread_count().
//   ebr       random shard counts (incl. 0 and past kEbrMaxShards) and random
//             enter/exit/retire/collect sequences: nothing freed while a reader
//             that predates its retire is pinned, everything freed exactly once.
//
// CLI: --count N sweeps seeds 1..N (default 200); --seed N replays one
// (and always runs the scheduler phase).

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_budget.h"
#include "bolt/bolt_ebr.h"
#include "bolt/bolt_limits.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_topology.h"

namespace {

uint64_t g_seed = 0;
const char* g_phase = "";

#define CHK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "[FAIL] seed=%llu phase=%s %s:%d %s\n", \
                 (unsigned long long)g_seed, g_phase, __FILE__, __LINE__, #cond); \
    std::exit(1); } } while (0)

struct Rng { uint64_t s; };

uint64_t rng_next(Rng* r) noexcept {
    r->s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = r->s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
Rng rng_seed(uint64_t seed) noexcept { Rng r{seed * 0xD1B54A32D192ED03ULL}; (void)rng_next(&r); return r; }
uint32_t below(Rng* r, uint32_t n) noexcept {
    return n == 0 ? 0 : static_cast<uint32_t>(((rng_next(r) & 0xFFFFFFFFULL) * n) >> 32);
}
bool chance(Rng* r, uint32_t pct) noexcept { return below(r, 100) < pct; }

// ---------------------------------------------------------------------------
// limits
// ---------------------------------------------------------------------------

constexpr const char* kPrimary = "BOLT_FUZZ_L8_WORKERS";
constexpr const char* kAlias   = "BOLT_FUZZ_L8_WORKERS_ALIAS";
constexpr uint32_t kStrCap = 48;

// Reference parser, written independently of limits_parse_u64.
bool ref_parse(const char* s, uint64_t* out) noexcept {
    const size_t n = std::strlen(s);
    if (n == 0) return false;
    unsigned __int128 v = 0;
    for (size_t i = 0; i < n; ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        v = v * 10u + static_cast<unsigned>(s[i] - '0');
        if (v > UINT64_MAX) return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

void put_u64(char* buf, uint64_t v) noexcept {
    std::snprintf(buf, kStrCap, "%llu", static_cast<unsigned long long>(v));
}

// kind: 0 unset, 1 empty, else a string near the interesting values.
void gen_value(Rng* r, uint64_t lo, uint64_t hi, char* buf, bool* set) noexcept {
    *set = true;
    buf[0] = '\0';
    switch (below(r, 12)) {
        case 0: *set = false; return;
        case 1: return;
        case 2: put_u64(buf, lo); return;
        case 3: put_u64(buf, hi); return;
        case 4: put_u64(buf, hi + 1u); return;          // wraps to 0 at UINT64_MAX
        case 5: put_u64(buf, lo - 1u); return;
        case 6: put_u64(buf, lo + (hi > lo ? rng_next(r) % (hi - lo + 1u) : 0u)); return;
        case 7: put_u64(buf, rng_next(r)); return;
        case 8: std::snprintf(buf, kStrCap, "18446744073709551616"); return;  // 2^64
        case 9: {                                        // leading zeros
            char t[kStrCap];
            put_u64(t, below(r, 5000));
            std::snprintf(buf, kStrCap, "000%s", t);
            return;
        }
        case 10: {                                       // digits with one junk byte
            static const char junk[] = " -+x.\t,e";
            put_u64(buf, below(r, 9000));
            const uint32_t n = static_cast<uint32_t>(std::strlen(buf));
            const uint32_t pos = below(r, n + 1u);
            buf[pos] = junk[below(r, sizeof(junk) - 1u)];
            if (pos == n) buf[n + 1] = '\0';
            return;
        }
        default: {                                       // long digit run
            const uint32_t n = 18u + below(r, 20);
            for (uint32_t i = 0; i < n; ++i) buf[i] = static_cast<char>('0' + below(r, 10));
            buf[n] = '\0';
            return;
        }
    }
}

void set_env(const char* name, bool set, const char* v) noexcept {
    if (set) ::setenv(name, v, 1); else ::unsetenv(name);
}

// Check one row of `t` after limits_resolve against the reference.
void check_row(const bolt::LimitTable& t, uint32_t i, const char* primary_name,
               bool pset, const char* pval, const char* alias_name,
               bool aset, const char* aval, bool resolve_ok) {
    const bolt::LimitDesc& d = t.descs[i];
    const uint64_t got = bolt::limits_value(t, i);
    const bolt::LimitSource src = bolt::limits_source(t, i);
    const bool use_primary = pset && pval[0] != '\0';
    const bool use_alias = !use_primary && alias_name != nullptr && aset && aval[0] != '\0';
    if (d.kind == bolt::LimitKind::kInvariant || (!use_primary && !use_alias)) {
        CHK(src == bolt::LimitSource::kDefault);
        CHK(got == d.def);
        return;
    }
    const char* raw = use_primary ? pval : aval;
    const char* var = use_primary ? primary_name : alias_name;
    uint64_t want = 0;
    if (ref_parse(raw, &want) && want >= d.min && want <= d.max) {
        CHK(src == bolt::LimitSource::kEnv);
        CHK(got == want);
        return;
    }
    CHK(src == bolt::LimitSource::kInvalid);
    CHK(got == d.def);
    CHK(!resolve_ok);
    char needle[128];
    std::snprintf(needle, sizeof(needle), "%s=%s is invalid", var, raw);
    CHK(t.error[0] != '\0');
    // The first invalid row owns the message; this fuzzer has one mutable row.
    CHK(std::strstr(t.error, needle) != nullptr);
}

void fuzz_limits_synthetic(Rng* r) {
    g_phase = "limits.synthetic";
    uint64_t lo = rng_next(r) % 5000u;
    uint64_t hi = lo + rng_next(r) % 10000u;
    if (chance(r, 10)) { lo = 0; hi = UINT64_MAX; }
    if (chance(r, 10)) { hi = UINT64_MAX; }
    const uint64_t def = lo + (hi > lo ? rng_next(r) % (hi - lo + 1u) : 0u);
    const bool alias = chance(r, 60);
    const bolt::LimitKind kind = chance(r, 10) ? bolt::LimitKind::kInvariant
                                               : bolt::LimitKind::kFixedAtAlloc;
    const bolt::LimitDesc desc{"workers", kind, "threads", def, lo, hi,
                               kPrimary, alias ? kAlias : nullptr, "fuzz"};
    std::atomic<uint64_t> values[1];
    std::atomic<uint8_t> sources[1];
    bolt::LimitTable t{"fuzz", &desc, 1, values, sources, {}};

    char pv[kStrCap], av[kStrCap];
    bool ps = false, as = false;
    gen_value(r, lo, hi, pv, &ps);
    gen_value(r, lo, hi, av, &as);
    set_env(kPrimary, ps, pv);
    set_env(kAlias, as, av);
    const bool ok = bolt::limits_resolve(&t);
    check_row(t, 0, kPrimary, ps, pv, alias ? kAlias : nullptr, as, av, ok);
    ::unsetenv(kPrimary);
    ::unsetenv(kAlias);
}

// The real bolt.workers row, re-resolved into scratch storage so the
// process-wide table (resolved once at startup) is never disturbed.
void fuzz_limits_bolt_workers(Rng* r) {
    g_phase = "limits.bolt_workers";
    const bolt::LimitTable& real = bolt::bolt_limits();
    const int32_t wi = bolt::limits_index(real, "workers");
    CHK(wi >= 0);
    const bolt::LimitDesc& d = real.descs[wi];
    CHK(d.env != nullptr && std::strcmp(d.env, "BOLT_WORKERS") == 0);
    CHK(d.max == BOLT_MAX_WORKERS && d.min == 0u);
    std::atomic<uint64_t> values[1];
    std::atomic<uint8_t> sources[1];
    bolt::LimitTable t{"bolt", &d, 1, values, sources, {}};
    char pv[kStrCap];
    bool ps = false;
    gen_value(r, 0, BOLT_MAX_WORKERS, pv, &ps);
    set_env("BOLT_WORKERS", ps, pv);
    const bool ok = bolt::limits_resolve(&t);
    check_row(t, 0, "BOLT_WORKERS", ps, pv, nullptr, false, "", ok);
    ::unsetenv("BOLT_WORKERS");
}

// ---------------------------------------------------------------------------
// scheduler
// ---------------------------------------------------------------------------

struct RangeSeen {
    std::atomic<uint8_t>* hits;
    uint32_t n_workers;
    std::atomic<uint32_t> bad_tid;
    std::atomic<uint32_t> bad_range;
    uint32_t count;
};

void range_mark(void* user, uint32_t start, uint32_t end, uint32_t tid) noexcept {
    auto* s = static_cast<RangeSeen*>(user);
    if (tid >= s->n_workers) s->bad_tid.fetch_add(1, std::memory_order_relaxed);
    if (start >= end || end > s->count) { s->bad_range.fetch_add(1); return; }
    for (uint32_t i = start; i < end; ++i) s->hits[i].fetch_add(1, std::memory_order_relaxed);
}

uint32_t pick_workers(Rng* r) noexcept {
    const uint32_t k = below(r, 100);
    if (k < 55) return 1u + below(r, 8);
    if (k < 85) return 9u + below(r, 64);
    if (k < 95) return 65u + below(r, 136);                     // past the old 64 clamp
    return bolt::kMaxWorkers + 1u + below(r, 5000);             // refused
}

void run_ranges(Rng* r, bolt::Scheduler* sched, uint32_t n) {
    const uint32_t rounds = 1u + below(r, 3);
    for (uint32_t k = 0; k < rounds; ++k) {
        uint32_t count = below(r, 4) == 0 ? below(r, 64) : below(r, 200000);
        const uint32_t grain = chance(r, 20) ? 1u + below(r, 4)
                                             : 1u + below(r, 1u << (1u + below(r, 16)));
        if (grain <= 4u && count > 20000u) count = 20000u;      // bound the task count
        auto* hits = new std::atomic<uint8_t>[count == 0 ? 1 : count];
        for (uint32_t i = 0; i < count; ++i) hits[i].store(0, std::memory_order_relaxed);
        RangeSeen seen{hits, n, {0}, {0}, count};
        sched->submit_range(&range_mark, &seen, count, grain);
        sched->wait_all();
        CHK(seen.bad_tid.load() == 0u);
        CHK(seen.bad_range.load() == 0u);
        for (uint32_t i = 0; i < count; ++i) CHK(hits[i].load() == 1u);
        delete[] hits;
    }
}

void fuzz_scheduler(Rng* r) {
    g_phase = "scheduler";
    auto* sched = new bolt::Scheduler();
    const uint32_t cycles = 1u + below(r, 3);
    for (uint32_t c = 0; c < cycles; ++c) {
        const uint32_t n = pick_workers(r);
        bool ok;
        if (chance(r, 50)) {
            bolt::SchedulerConfig cfg{};
            cfg.num_workers = n;
            cfg.pin_workers = chance(r, 30);
            cfg.numa_bind = chance(r, 20);
            cfg.prefer_p_cores = chance(r, 50);
            ok = sched->init(cfg);
        } else {
            ok = sched->init(n);
        }
        if (n > bolt::kMaxWorkers) {
            CHK(!ok);
            CHK(sched->thread_count() == 0u);
            sched->shutdown();                                   // safe after refusal
            continue;
        }
        CHK(ok);
        CHK(sched->thread_count() == n);
        for (uint32_t w = 0; w < n; ++w) CHK(sched->worker_arena(w) != nullptr);
        run_ranges(r, sched, n);
        sched->shutdown();
    }
    delete sched;
}

// ---------------------------------------------------------------------------
// ebr (single-threaded op sequences; the concurrent protocol has its own tests)
// ---------------------------------------------------------------------------

constexpr uint32_t kMaxObjs = 4096;

struct EbrObj {
    uint64_t retire_seq;
    uint32_t freed;
    uint32_t accepted;
};

struct EbrModel {
    EbrObj objs[kMaxObjs];
    uint64_t* enter_seq;         // per shard; 0 = unpinned
    uint32_t* pinned;            // shard ids currently pinned
    uint32_t n_pinned;
    uint64_t seq;
    uint32_t violations;
};

EbrModel* g_model = nullptr;

void ebr_free_obj(void* p) {
    EbrObj* o = static_cast<EbrObj*>(p);
    EbrModel* m = g_model;
    o->freed++;
    for (uint32_t i = 0; i < m->n_pinned; ++i) {
        if (m->enter_seq[m->pinned[i]] < o->retire_seq) m->violations++;
    }
}

uint32_t pick_shards(Rng* r) noexcept {
    const uint32_t k = below(r, 100);
    if (k < 5) return 0u;
    if (k < 10) return bolt::kEbrMaxShards + 1u + below(r, 64);
    if (k < 60) return 1u + below(r, 8);
    if (k < 90) return 1u + below(r, 256);
    return 1u + below(r, bolt::kEbrMaxShards);
}

void unpin(EbrModel* m, uint32_t slot_idx) noexcept {
    m->enter_seq[m->pinned[slot_idx]] = 0;
    m->pinned[slot_idx] = m->pinned[--m->n_pinned];
}

void fuzz_ebr(Rng* r) {
    g_phase = "ebr";
    auto* e = new bolt::Ebr();
    const uint32_t shards = pick_shards(r);
    const bool ok = bolt::ebr_init(e, shards);
    if (shards == 0u || shards > bolt::kEbrMaxShards) {
        CHK(!ok);
        CHK(e->num_shards == 0u && e->shards == nullptr);
        bolt::ebr_destroy(e);
        delete e;
        return;
    }
    CHK(ok && e->num_shards == shards && e->shards != nullptr);

    auto* m = new EbrModel();
    std::memset(m->objs, 0, sizeof(m->objs));
    m->enter_seq = new uint64_t[shards]();
    m->pinned = new uint32_t[shards];
    m->n_pinned = 0;
    m->seq = 1;
    m->violations = 0;
    g_model = m;

    // Pins cluster on a few shards so collisions happen at any width.
    const uint32_t hot = 1u + below(r, shards < 6 ? shards : 6);
    uint32_t n_objs = 0;
    const uint32_t ops = 200u + below(r, 3000);
    for (uint32_t op = 0; op < ops; ++op) {
        const uint32_t sid = chance(r, 70) ? below(r, hot) : below(r, shards);
        const uint32_t k = below(r, 100);
        if (k < 25) {
            if (m->enter_seq[sid] == 0) {
                bolt::ebr_enter(e, sid);
                m->enter_seq[sid] = ++m->seq;
                m->pinned[m->n_pinned++] = sid;
            }
        } else if (k < 45) {
            if (m->n_pinned > 0) {
                const uint32_t j = below(r, m->n_pinned);
                bolt::ebr_exit(e, m->pinned[j]);
                unpin(m, j);
            }
        } else if (k < 80) {
            if (n_objs < kMaxObjs) {
                EbrObj* o = &m->objs[n_objs++];
                o->retire_seq = ++m->seq;
                o->accepted = bolt::ebr_retire(e, sid, o, &ebr_free_obj) ? 1u : 0u;
            }
        } else {
            (void)bolt::ebr_try_advance_and_collect(e);
        }
        CHK(m->violations == 0u);
    }
    while (m->n_pinned > 0) { bolt::ebr_exit(e, m->pinned[0]); unpin(m, 0); }
    bolt::ebr_destroy(e);
    CHK(e->num_shards == 0u && e->shards == nullptr);
    CHK(m->violations == 0u);
    for (uint32_t i = 0; i < n_objs; ++i) CHK(m->objs[i].freed == m->objs[i].accepted);
    g_model = nullptr;
    delete[] m->pinned;
    delete[] m->enter_seq;
    delete m;
    delete e;
}

void check_topology() {
    g_phase = "topology";
    bolt::CpuTopology t{};
    CHK(bolt::bolt_detect_topology(&t));
    CHK(t.logical_cpus >= 1u && t.logical_cpus <= bolt::kTopologyMaxCpus);
    CHK(t.numa_nodes >= 1u);
    for (uint32_t i = 0; i < t.logical_cpus; ++i) CHK(t.cpu_to_node[i] < t.numa_nodes);
    CHK(bolt::bolt_auto_workers() >= 1u && bolt::bolt_auto_workers() <= bolt::kMaxWorkers);
}

void run_seed(uint64_t seed) {
    g_seed = seed;
    Rng r = rng_seed(seed);
    for (uint32_t i = 0; i < 8; ++i) fuzz_limits_synthetic(&r);
    fuzz_limits_bolt_workers(&r);
    fuzz_ebr(&r);
    if (seed % 4u == 1u) fuzz_scheduler(&r);   // thread spawn dominates cost
}

}  // namespace

int main(int argc, char** argv) {
    // Resolve the process tables before any setenv so they see the real env.
    (void)bolt::bolt_limits();
    ::unsetenv(kPrimary);
    ::unsetenv(kAlias);
    uint64_t count = 200, one = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--count") == 0) count = std::strtoull(argv[i + 1], nullptr, 10);
        else if (std::strcmp(argv[i], "--seed") == 0) one = std::strtoull(argv[i + 1], nullptr, 10);
    }
    const char* saved = std::getenv("BOLT_WORKERS");
    char saved_buf[64] = {0};
    if (saved != nullptr) std::snprintf(saved_buf, sizeof(saved_buf), "%s", saved);
    check_topology();
    if (one != 0) {
        run_seed(one);
        // --seed always exercises the scheduler too.
        Rng r = rng_seed(one ^ 0x5CEDu);
        fuzz_scheduler(&r);
    } else {
        for (uint64_t s = 1; s <= count; ++s) run_seed(s);
    }
    if (saved != nullptr) ::setenv("BOLT_WORKERS", saved_buf, 1);
    std::printf("fuzz_workers_kind_b: ok (%llu seeds)\n",
                static_cast<unsigned long long>(one != 0 ? 1 : count));
    return 0;
}
