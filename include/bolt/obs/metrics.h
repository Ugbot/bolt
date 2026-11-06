// bolt/obs/metrics.h — counters and fixed-bucket histograms for hot paths.
//
// Series are registered once, cold, and resolve to a cell index. A thread owns
// a private shard of cells, so an update is one relaxed add on a line no other
// core writes; scrape sums the shards. The run-time switch is one relaxed load.
// With BOLT_OBS_METRICS=0 every type is empty and every update compiles away.
//
// Gauges that a subsystem already tracks are read at scrape by a collector,
// so they cost the hot path nothing.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

#include "bolt/obs/obs_config.h"

namespace bolt::metrics {

enum class Format : std::uint8_t { Prometheus = 0, OpenMetrics };

// Log2 buckets. Latency records ns and renders seconds; Size records bytes.
struct Buckets {
    std::uint32_t first_shift;   // top of bucket 0 is 2^first_shift
    std::uint32_t count;         // finite buckets; +Inf is implied
    double scale;                // rendered bound and sum multiplier
};
inline constexpr Buckets kLatency{10, 27, 1e-9};   // ~1 us .. ~69 s
inline constexpr Buckets kSize{10, 24, 1.0};       // 1 KiB .. 8 GiB
inline constexpr Buckets kCount{0, 24, 1.0};       // 1 .. 8M

class Writer {
public:
    Writer(std::string* out, Format format) noexcept : out_(out), format_(format) {}
    // Starts a family; "counter" names end in _total, as OpenMetrics requires.
    void family(std::string_view name, const char* type, std::string_view help);
    void sample(std::string_view labels, double value);
    void sample_u64(std::string_view labels, std::uint64_t value);
    void raw(std::string_view text);
private:
    void head(std::string_view labels);
    std::string* out_;
    Format format_;
    std::string name_;
    bool om_counter_ = false;
};

using Collector = void (*)(void* ctx, Writer& w);

// Rewrite a Prometheus 0.0.4 body (a legacy renderer) for OpenMetrics.
void to_openmetrics(std::string_view prometheus_text, std::string* out);

#if BOLT_OBS_METRICS

using Cell = std::atomic<std::uint64_t>;

namespace detail {
extern constinit std::atomic<std::uint8_t> g_enabled;
extern constinit thread_local Cell* t_shard;
Cell* attach_shard() noexcept;
inline Cell* shard() noexcept {
    Cell* s = t_shard;
    return s != nullptr ? s : attach_shard();
}
}  // namespace detail

inline bool enabled() noexcept { return detail::g_enabled.load(std::memory_order_relaxed) != 0; }
void set_enabled(bool on) noexcept;

class Counter {
public:
    constexpr Counter() noexcept = default;
    explicit constexpr Counter(std::uint32_t cell) noexcept : cell_(cell) {}
    void inc(std::uint64_t n = 1) const noexcept {
        if (!enabled()) return;
        detail::shard()[cell_].fetch_add(n, std::memory_order_relaxed);
    }
private:
    std::uint32_t cell_ = 0;
};

// Summed across shards as signed: add on one thread, subtract on another.
class Gauge {
public:
    constexpr Gauge() noexcept = default;
    explicit constexpr Gauge(std::uint32_t cell) noexcept : cell_(cell) {}
    void add(std::int64_t n) const noexcept {
        if (!enabled()) return;
        detail::shard()[cell_].fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);
    }
private:
    std::uint32_t cell_ = 0;
};

class Histogram {
public:
    constexpr Histogram() noexcept = default;
    constexpr Histogram(std::uint32_t base, std::uint32_t shift, std::uint32_t count) noexcept
        : base_(base), shift_(shift), count_(count) {}
    void observe(std::uint64_t v) const noexcept {
        if (!enabled()) return;
        const std::uint32_t top = 64u - static_cast<std::uint32_t>(__builtin_clzll(v | 1u));
        std::uint32_t b = top > shift_ ? top - shift_ : 0u;
        if (v == (1ull << (top - 1)) && b != 0) --b;   // 2^k sits in the bucket it tops
        if (b > count_) b = count_;
        Cell* s = detail::shard();
        s[base_ + b].fetch_add(1, std::memory_order_relaxed);
        s[base_ + count_ + 1].fetch_add(v, std::memory_order_relaxed);
    }
private:
    std::uint32_t base_ = 0;
    std::uint32_t shift_ = 0;
    std::uint32_t count_ = 0;
};

// labels: `a="b",c="d"` or empty. Past the series or cell budget a series
// shares the overflow cell and bolt_metrics_series_overflow_total counts it.
Counter counter(std::string_view name, std::string_view help, std::string_view labels = {}) noexcept;
Gauge gauge(std::string_view name, std::string_view help, std::string_view labels = {}) noexcept;
Histogram histogram(std::string_view name, std::string_view help, std::string_view labels,
                    const Buckets& buckets) noexcept;

void add_collector(Collector fn, void* ctx) noexcept;
void render(std::string* out, Format format);

// Read one series back (tests, bundles); 0 when unknown.
std::uint64_t counter_value(Counter c) noexcept;

// A bounded, lock-free string -> value table for label values that arrive at
// run time. Past `capacity` a key maps to the overflow slot.
class LabelTable {
public:
    using Build = void (*)(std::string_view key, void* slot, void* ctx);
    LabelTable(std::uint32_t capacity, std::uint32_t slot_bytes, Build build, void* ctx) noexcept;
    void* find_or_add(std::string_view key) noexcept;
    std::uint32_t size() const noexcept { return size_.load(std::memory_order_acquire); }
private:
    struct Entry;
    Entry* add(std::string_view key, std::uint64_t hash) noexcept;
    std::atomic<Entry*>* table_ = nullptr;
    std::uint32_t mask_ = 0;
    std::uint32_t capacity_;
    std::uint32_t slot_bytes_;
    Build build_;
    void* ctx_;
    std::atomic<std::uint32_t> size_{0};
    std::atomic<Entry*> overflow_{nullptr};
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
};

#else  // BOLT_OBS_METRICS

inline constexpr bool enabled() noexcept { return false; }
inline void set_enabled(bool) noexcept {}

class Counter { public: constexpr Counter() noexcept = default; void inc(std::uint64_t = 1) const noexcept {} };
class Gauge { public: constexpr Gauge() noexcept = default; void add(std::int64_t) const noexcept {} };
class Histogram { public: constexpr Histogram() noexcept = default; void observe(std::uint64_t) const noexcept {} };

inline Counter counter(std::string_view, std::string_view, std::string_view = {}) noexcept { return {}; }
inline Gauge gauge(std::string_view, std::string_view, std::string_view = {}) noexcept { return {}; }
inline Histogram histogram(std::string_view, std::string_view, std::string_view, const Buckets&) noexcept { return {}; }
inline void add_collector(Collector, void*) noexcept {}
inline void render(std::string*, Format) {}
inline std::uint64_t counter_value(Counter) noexcept { return 0; }

#endif  // BOLT_OBS_METRICS

}  // namespace bolt::metrics
