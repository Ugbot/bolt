// bolt_limits.h — the runtime Limits registry.
//
// Every capacity has a kind (see the Gestalt2 limits review):
//   A kInvariant   — fixed by a format, wire, SIMD width or algorithm. Reported,
//                    never set. min == max == default.
//   B kFixedAtAlloc — sized once at start/open from config (pools, rings,
//                    workers, caches). Settable before first use.
//   C kBudget      — a budget that data-sized structures grow under. Settable.
//
// Each repo declares ONE X-macro table and instantiates it with
// BOLT_LIMITS_TABLE. The table resolves its environment overrides once, on
// first access, and registers itself here so a host (gestaltd) can report
// every value with its source (admin.config.get) and refuse to start on an
// out-of-range override.
//
//   #define MYREPO_LIMITS(X) \
//     X(max_widgets, kFixedAtAlloc, "count", 1024, 1, 1u << 20, \
//       "MYREPO_MAX_WIDGETS", nullptr, "widgets per pool")
//   BOLT_LIMITS_TABLE(myrepo_limits, "myrepo", MYREPO_LIMITS)
//   ... myrepo_limits_value(myrepo_limits_id::max_widgets)
//
// RULES: No exceptions, no allocation. Environment parsing happens once per
// table, at first access (never on a hot path).

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bolt {

enum class LimitKind : uint8_t {
    kInvariant   = 0,   // A
    kFixedAtAlloc = 1,  // B
    kBudget      = 2,   // C
};

enum class LimitSource : uint8_t {
    kDefault = 0,
    kEnv     = 1,
    kSet     = 2,   // programmatic / CLI / config file
};

inline const char* limit_kind_name(LimitKind k) noexcept {
    switch (k) {
        case LimitKind::kInvariant:    return "A";
        case LimitKind::kFixedAtAlloc: return "B";
        case LimitKind::kBudget:       return "C";
    }
    return "?";
}

inline const char* limit_source_name(LimitSource s) noexcept {
    switch (s) {
        case LimitSource::kDefault: return "default";
        case LimitSource::kEnv:     return "env";
        case LimitSource::kSet:     return "set";
    }
    return "?";
}

struct LimitDesc {
    const char* name;
    LimitKind   kind;
    const char* unit;
    uint64_t    def;
    uint64_t    min;
    uint64_t    max;
    const char* env;        // nullptr = no environment override
    const char* env_alias;  // legacy name, read when `env` is unset
    const char* doc;
};

inline constexpr uint32_t kLimitsErrorBytes = 256;

struct LimitTable {
    const char*            owner;
    const LimitDesc*       descs;
    uint32_t               n;
    std::atomic<uint64_t>* values;
    std::atomic<uint8_t>*  sources;
    char                   error[kLimitsErrorBytes];   // first resolve error
};

// Registry capacity: one table per repo, generous headroom.
inline constexpr uint32_t kLimitsMaxTables = 64;

namespace detail {
struct LimitsRegistry {
    std::atomic<LimitTable*> tables[kLimitsMaxTables];
    std::atomic<uint32_t>    count;
    std::atomic<uint32_t>    dropped;   // registrations refused (full)
};
inline LimitsRegistry& limits_registry() noexcept {
    static LimitsRegistry r{};
    return r;
}
}  // namespace detail

// Parse a base-10 unsigned integer, whole string. false on empty, junk or
// overflow.
inline bool limits_parse_u64(const char* s, uint64_t* out) noexcept {
    assert(out != nullptr);
    if (s == nullptr || *s == '\0') return false;
    uint64_t v = 0;
    uint32_t i = 0;
    for (; s[i] != '\0' && i < 32; ++i) {
        const char c = s[i];
        if (c < '0' || c > '9') return false;
        const uint64_t d = static_cast<uint64_t>(c - '0');
        if (v > (UINT64_MAX - d) / 10u) return false;
        v = v * 10u + d;
    }
    if (s[i] != '\0') return false;   // > 32 digits cannot be a u64
    *out = v;
    return true;
}

inline bool limits_in_range(const LimitDesc& d, uint64_t v) noexcept {
    return v >= d.min && v <= d.max;
}

// Resolve a table's environment overrides. Records the first error in
// table->error and keeps that entry at its default. Returns false on error.
inline bool limits_resolve(LimitTable* t) noexcept {
    assert(t != nullptr && t->descs != nullptr);
    assert(t->n > 0);
    bool ok = true;
    for (uint32_t i = 0; i < t->n; ++i) {
        const LimitDesc& d = t->descs[i];
        assert(d.min <= d.def && d.def <= d.max);
        t->values[i].store(d.def, std::memory_order_relaxed);
        t->sources[i].store(static_cast<uint8_t>(LimitSource::kDefault),
                            std::memory_order_relaxed);
        if (d.kind == LimitKind::kInvariant) continue;
        // An empty variable counts as unset.
        const char* var = d.env;
        const char* raw = d.env != nullptr ? std::getenv(d.env) : nullptr;
        if ((raw == nullptr || *raw == '\0') && d.env_alias != nullptr) {
            raw = std::getenv(d.env_alias);
            var = d.env_alias;
        }
        if (raw == nullptr || *raw == '\0') continue;
        uint64_t v = 0;
        if (!limits_parse_u64(raw, &v) || !limits_in_range(d, v)) {
            if (ok) {
                std::snprintf(t->error, sizeof(t->error),
                              "%s=%s is invalid for limit %s.%s: expected an "
                              "integer in [%llu, %llu] (%s)",
                              var, raw, t->owner, d.name,
                              static_cast<unsigned long long>(d.min),
                              static_cast<unsigned long long>(d.max), d.unit);
            }
            ok = false;
            continue;
        }
        t->values[i].store(v, std::memory_order_relaxed);
        t->sources[i].store(static_cast<uint8_t>(LimitSource::kEnv),
                            std::memory_order_relaxed);
    }
    return ok;
}

// Register a resolved table. Idempotent per pointer. false when full.
inline bool limits_register(LimitTable* t) noexcept {
    assert(t != nullptr);
    detail::LimitsRegistry& r = detail::limits_registry();
    const uint32_t n = r.count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n && i < kLimitsMaxTables; ++i) {
        if (r.tables[i].load(std::memory_order_acquire) == t) return true;
    }
    const uint32_t slot = r.count.fetch_add(1, std::memory_order_acq_rel);
    if (slot >= kLimitsMaxTables) {
        r.count.fetch_sub(1, std::memory_order_acq_rel);
        r.dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    r.tables[slot].store(t, std::memory_order_release);
    return true;
}

inline uint32_t limits_table_count() noexcept {
    const uint32_t n = detail::limits_registry().count.load(std::memory_order_acquire);
    return n < kLimitsMaxTables ? n : kLimitsMaxTables;
}

// nullptr while a concurrent registration is mid-publish.
inline const LimitTable* limits_table_at(uint32_t i) noexcept {
    assert(i < kLimitsMaxTables);
    return detail::limits_registry().tables[i].load(std::memory_order_acquire);
}

// First resolve error across every registered table, or nullptr.
inline const char* limits_first_error() noexcept {
    if (detail::limits_registry().dropped.load(std::memory_order_relaxed) != 0) {
        return "limits registry full: raise bolt::kLimitsMaxTables";
    }
    const uint32_t n = limits_table_count();
    for (uint32_t i = 0; i < n; ++i) {
        const LimitTable* t = limits_table_at(i);
        if (t != nullptr && t->error[0] != '\0') return t->error;
    }
    return nullptr;
}

inline int32_t limits_index(const LimitTable& t, const char* name) noexcept {
    assert(name != nullptr);
    for (uint32_t i = 0; i < t.n; ++i) {
        if (std::strcmp(t.descs[i].name, name) == 0) return static_cast<int32_t>(i);
    }
    return -1;
}

// Find a limit by name across every registered table.
inline bool limits_find(const char* name, const LimitTable** out_t,
                        uint32_t* out_i) noexcept {
    assert(name != nullptr && out_t != nullptr && out_i != nullptr);
    const uint32_t n = limits_table_count();
    for (uint32_t k = 0; k < n; ++k) {
        const LimitTable* t = limits_table_at(k);
        if (t == nullptr) continue;
        const int32_t i = limits_index(*t, name);
        if (i >= 0) { *out_t = t; *out_i = static_cast<uint32_t>(i); return true; }
    }
    return false;
}

// Programmatic override (CLI/config). Invariants cannot be set; the value
// must be in range. Writes a reason into err on failure.
inline bool limits_set(LimitTable* t, uint32_t i, uint64_t v,
                       char* err, size_t err_cap) noexcept {
    assert(t != nullptr && i < t->n);
    assert(err == nullptr || err_cap > 0);
    const LimitDesc& d = t->descs[i];
    if (d.kind == LimitKind::kInvariant || !limits_in_range(d, v)) {
        if (err != nullptr) {
            std::snprintf(err, err_cap, "%s.%s: %llu is %s [%llu, %llu]",
                          t->owner, d.name, static_cast<unsigned long long>(v),
                          d.kind == LimitKind::kInvariant
                              ? "a compile-time invariant, fixed at"
                              : "outside",
                          static_cast<unsigned long long>(d.min),
                          static_cast<unsigned long long>(d.max));
        }
        return false;
    }
    t->values[i].store(v, std::memory_order_relaxed);
    t->sources[i].store(static_cast<uint8_t>(LimitSource::kSet),
                        std::memory_order_relaxed);
    return true;
}

inline uint64_t limits_value(const LimitTable& t, uint32_t i) noexcept {
    assert(i < t.n);
    return t.values[i].load(std::memory_order_relaxed);
}

inline LimitSource limits_source(const LimitTable& t, uint32_t i) noexcept {
    assert(i < t.n);
    return static_cast<LimitSource>(t.sources[i].load(std::memory_order_relaxed));
}

// The environment variable that raises a named limit, or nullptr.
inline const char* limits_env_for(const char* name) noexcept {
    const LimitTable* t = nullptr;
    uint32_t i = 0;
    if (name == nullptr || !limits_find(name, &t, &i)) return nullptr;
    return t->descs[i].env;
}

}  // namespace bolt

#define BOLT_LIMITS_X_ENUM(name, kind, unit, def, mn, mx, env, alias, doc) name,
#define BOLT_LIMITS_X_DESC(name, kind, unit, def, mn, mx, env, alias, doc) \
    ::bolt::LimitDesc{#name, ::bolt::LimitKind::kind, unit,                 \
                      static_cast<uint64_t>(def), static_cast<uint64_t>(mn), \
                      static_cast<uint64_t>(mx), env, alias, doc},

// Defines `enum class <fn>_id`, `bolt::LimitTable& <fn>()` (resolves and
// registers on first call) and `uint64_t <fn>_value(<fn>_id)`.
#define BOLT_LIMITS_TABLE(fn, owner_str, LIST)                                  \
    enum class fn##_id : uint32_t { LIST(BOLT_LIMITS_X_ENUM) kCount };          \
    inline ::bolt::LimitTable& fn() noexcept {                                  \
        static const ::bolt::LimitDesc descs[] = {LIST(BOLT_LIMITS_X_DESC)};    \
        static_assert(sizeof(descs) / sizeof(descs[0]) ==                       \
                      static_cast<size_t>(fn##_id::kCount));                    \
        static std::atomic<uint64_t> values[static_cast<size_t>(fn##_id::kCount)]; \
        static std::atomic<uint8_t> sources[static_cast<size_t>(fn##_id::kCount)]; \
        static ::bolt::LimitTable table{                                        \
            owner_str, descs, static_cast<uint32_t>(fn##_id::kCount), values,   \
            sources, {}};                                                        \
        static const bool once = [] {                                           \
            (void)::bolt::limits_resolve(&table);                               \
            return ::bolt::limits_register(&table);                             \
        }();                                                                     \
        (void)once;                                                             \
        return table;                                                           \
    }                                                                           \
    inline uint64_t fn##_value(fn##_id id) noexcept {                           \
        return ::bolt::limits_value(fn(), static_cast<uint32_t>(id));           \
    }
