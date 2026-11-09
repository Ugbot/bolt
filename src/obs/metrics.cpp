// metrics.cpp — series registry, shards, exposition.
#include "bolt/obs/metrics.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bolt::metrics {

void Writer::family(std::string_view name, const char* type, std::string_view help) {
    assert(out_ != nullptr && type != nullptr);
    assert(!name.empty());
    name_.assign(name);
    om_counter_ = false;
    std::string_view base = name;
    if (format_ == Format::OpenMetrics && std::strcmp(type, "counter") == 0 && base.size() > 6 &&
        base.substr(base.size() - 6) == "_total") {
        base.remove_suffix(6);
        om_counter_ = true;
    }
    out_->append("# HELP ").append(base).append(" ").append(help).append("\n# TYPE ").append(base)
        .append(" ").append(type).append("\n");
}

void Writer::head(std::string_view labels) {
    out_->append(name_);
    if (!labels.empty()) out_->append("{").append(labels).append("}");
    out_->push_back(' ');
}

void Writer::sample(std::string_view labels, double value) {
    head(labels);
    char b[40];
    std::snprintf(b, sizeof(b), "%.17g\n", value);
    out_->append(b);
}

void Writer::sample_u64(std::string_view labels, std::uint64_t value) {
    head(labels);
    out_->append(std::to_string(value)).push_back('\n');
}

void Writer::raw(std::string_view text) { out_->append(text); }

void to_openmetrics(std::string_view text, std::string* out) {
    assert(out != nullptr);
    std::size_t pos = 0;
    std::string pending_help;
    while (pos < text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        std::string_view line = text.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.substr(0, 7) == "# HELP ") { pending_help.assign(line); continue; }
        if (line.substr(0, 7) == "# TYPE ") {
            std::string_view rest = line.substr(7);
            const std::size_t sp = rest.find(' ');
            std::string_view name = rest.substr(0, sp);
            const std::string_view type = rest.substr(sp + 1);
            const bool strip = type == "counter" && name.size() > 6 && name.substr(name.size() - 6) == "_total";
            if (strip) {
                const std::string_view base = name.substr(0, name.size() - 6);
                if (!pending_help.empty()) {
                    out->append("# HELP ").append(base).append(pending_help.substr(7 + name.size())).push_back('\n');
                }
                out->append("# TYPE ").append(base).append(" counter\n");
            } else {
                if (!pending_help.empty()) out->append(pending_help).push_back('\n');
                out->append(line).push_back('\n');
            }
            pending_help.clear();
            continue;
        }
        out->append(line).push_back('\n');
    }
}

}  // namespace bolt::metrics

#if BOLT_OBS_METRICS

#include <mutex>
#include <new>
#include <vector>

#include "bolt/obs/obs_limits.h"

namespace bolt::metrics {

namespace detail {
constinit std::atomic<std::uint8_t> g_enabled{1};
constinit thread_local Cell* t_shard = nullptr;
}  // namespace detail

namespace {

enum class Kind : std::uint8_t { Counter, Gauge, Histogram };

struct Series {
    std::string name;
    std::string help;
    std::string labels;
    Kind kind;
    std::uint32_t cell;
    Buckets buckets;
};

struct Registry {
    std::mutex mu;
    std::vector<Series> series;
    std::vector<std::pair<Collector, void*>> collectors;
    std::uint32_t next_cell = 2;   // 0: unset handles, 1: overflow
    std::uint64_t overflow = 0;
};

Registry& registry() {
    static Registry* r = new Registry();
    return *r;
}

struct Shards {
    std::atomic<Cell*>* table;
    std::uint32_t count;
    std::uint32_t cells;
};

Shards& shards() {
    static Shards s = [] {
        Shards v{};
        v.count = static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_metrics_shards));
        v.cells = static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_metrics_cells));
        v.table = new std::atomic<Cell*>[v.count]();
        detail::g_enabled.store(obs::obs_limits_value(obs::obs_limits_id::obs_metrics_enabled) != 0 ? 1 : 0,
                                std::memory_order_relaxed);
        return v;
    }();
    return s;
}

constinit std::atomic<std::uint32_t> g_next_shard{0};

std::uint32_t reserve_cells(Registry& r, std::uint32_t n) noexcept {
    const std::uint32_t max_series =
        static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_metrics_series));
    if (r.next_cell + n > shards().cells || r.series.size() >= max_series) {
        ++r.overflow;
        return 1;
    }
    const std::uint32_t c = r.next_cell;
    r.next_cell += n;
    return c;
}

std::uint32_t add_series(std::string_view name, std::string_view help, std::string_view labels, Kind kind,
                         std::uint32_t cells, const Buckets& b) noexcept {
    assert(!name.empty());
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    for (const Series& s : r.series) {
        if (s.name == name && s.labels == labels) return s.cell;
    }
    const std::uint32_t cell = reserve_cells(r, cells);
    if (cell == 1) return cell;
    r.series.push_back(Series{std::string(name), std::string(help), std::string(labels), kind, cell, b});
    return cell;
}

std::uint64_t sum_cell(const Shards& s, std::uint32_t cell) noexcept {
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < s.count; ++i) {
        Cell* block = s.table[i].load(std::memory_order_acquire);
        if (block != nullptr) total += block[cell].load(std::memory_order_relaxed);
    }
    return total;
}

void bound_text(char* b, std::size_t cap, const Buckets& k, std::uint32_t i) noexcept {
    std::snprintf(b, cap, "%.9g", static_cast<double>(1ull << (k.first_shift + i)) * k.scale);
}

void render_histogram(std::string* out, const Series& s, const Shards& sh) {
    char le[32];
    std::string lab = s.labels.empty() ? std::string() : s.labels + ",";
    std::uint64_t cum = 0;
    for (std::uint32_t i = 0; i <= s.buckets.count; ++i) {
        cum += sum_cell(sh, s.cell + i);
        if (i == s.buckets.count) std::snprintf(le, sizeof(le), "+Inf");
        else bound_text(le, sizeof(le), s.buckets, i);
        out->append(s.name).append("_bucket{").append(lab).append("le=\"").append(le).append("\"} ")
            .append(std::to_string(cum)).push_back('\n');
    }
    const std::string braces = s.labels.empty() ? std::string() : "{" + s.labels + "}";
    char sum[40];
    std::snprintf(sum, sizeof(sum), "%.17g", static_cast<double>(sum_cell(sh, s.cell + s.buckets.count + 1)) * s.buckets.scale);
    out->append(s.name).append("_sum").append(braces).append(" ").append(sum).push_back('\n');
    out->append(s.name).append("_count").append(braces).append(" ").append(std::to_string(cum)).push_back('\n');
}

const char* type_name(Kind k) noexcept {
    return k == Kind::Counter ? "counter" : k == Kind::Gauge ? "gauge" : "histogram";
}

}  // namespace

namespace detail {

Cell* attach_shard() noexcept {
    Shards& s = shards();
    const std::uint32_t i = g_next_shard.fetch_add(1, std::memory_order_relaxed) % s.count;
    Cell* block = s.table[i].load(std::memory_order_acquire);
    if (block == nullptr) {
        auto* fresh = static_cast<Cell*>(std::calloc(s.cells, sizeof(Cell)));
        if (fresh == nullptr) std::abort();
        if (s.table[i].compare_exchange_strong(block, fresh, std::memory_order_acq_rel)) block = fresh;
        else std::free(fresh);
    }
    t_shard = block;
    return block;
}

}  // namespace detail

void set_enabled(bool on) noexcept {
    (void)shards();
    detail::g_enabled.store(on ? 1 : 0, std::memory_order_relaxed);
}

Counter counter(std::string_view name, std::string_view help, std::string_view labels) noexcept {
    return Counter(add_series(name, help, labels, Kind::Counter, 1, Buckets{}));
}

Gauge gauge(std::string_view name, std::string_view help, std::string_view labels) noexcept {
    return Gauge(add_series(name, help, labels, Kind::Gauge, 1, Buckets{}));
}

Histogram histogram(std::string_view name, std::string_view help, std::string_view labels, const Buckets& b) noexcept {
    assert(b.count > 0 && b.first_shift + b.count < 63);
    const std::uint32_t cell = add_series(name, help, labels, Kind::Histogram, b.count + 2, b);
    if (cell == 1) return Histogram(1, b.first_shift, 0);
    return Histogram(cell, b.first_shift, b.count);
}

void add_collector(Collector fn, void* ctx) noexcept {
    assert(fn != nullptr);
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    r.collectors.emplace_back(fn, ctx);
}

std::uint64_t counter_value(Counter c) noexcept {
    std::uint32_t cell;
    static_assert(sizeof(c) == sizeof(cell));
    std::memcpy(&cell, &c, sizeof(cell));
    return sum_cell(shards(), cell);
}

void render(std::string* out, Format format) {
    assert(out != nullptr);
    Registry& r = registry();
    const Shards& sh = shards();
    Writer w(out, format);
    std::lock_guard<std::mutex> lock(r.mu);
    // Group by family: a family's series may have registered at different times.
    std::vector<std::uint8_t> done(r.series.size(), 0);
    for (std::size_t i = 0; i < r.series.size(); ++i) {
        if (done[i] != 0) continue;
        const Series& head = r.series[i];
        w.family(head.name, type_name(head.kind), head.help);
        for (std::size_t j = i; j < r.series.size(); ++j) {
            const Series& s = r.series[j];
            if (s.name != head.name) continue;
            done[j] = 1;
            if (s.kind == Kind::Histogram) { render_histogram(out, s, sh); continue; }
            const std::uint64_t v = sum_cell(sh, s.cell);
            if (s.kind == Kind::Gauge) w.sample(s.labels, static_cast<double>(static_cast<std::int64_t>(v)));
            else w.sample_u64(s.labels, v);
        }
    }
    w.family("bolt_metrics_series_overflow_total", "counter", "series refused past the budget; raise obs_metrics_series or obs_metrics_cells");
    w.sample_u64({}, r.overflow);
    for (const auto& [fn, ctx] : r.collectors) fn(ctx, w);
}

struct LabelTable::Entry {
    std::uint64_t hash;
    std::uint32_t key_len;
    char* key;
    void* slot;
};

LabelTable::LabelTable(std::uint32_t capacity, std::uint32_t slot_bytes, Build build, void* ctx) noexcept
    : capacity_(capacity), slot_bytes_(slot_bytes), build_(build), ctx_(ctx) {
    assert(capacity > 0 && build != nullptr);
    std::uint32_t n = 16;
    while (n < capacity * 2) n <<= 1;
    table_ = new (std::nothrow) std::atomic<Entry*>[n]();
    if (table_ == nullptr) std::abort();
    mask_ = n - 1;
}

namespace {
std::uint64_t fnv1a(std::string_view s) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (char c : s) { h ^= static_cast<unsigned char>(c); h *= 0x100000001b3ull; }
    return h;
}
}  // namespace

void* LabelTable::find_or_add(std::string_view key) noexcept {
    const std::uint64_t h = fnv1a(key);
    for (std::uint32_t i = 0, slot = static_cast<std::uint32_t>(h) & mask_; i <= mask_; ++i, slot = (slot + 1) & mask_) {
        Entry* e = table_[slot].load(std::memory_order_acquire);
        if (e == nullptr) break;
        if (e->hash == h && e->key_len == key.size() && std::memcmp(e->key, key.data(), key.size()) == 0) return e->slot;
    }
    return add(key, h)->slot;
}

LabelTable::Entry* LabelTable::add(std::string_view key, std::uint64_t h) noexcept {
    while (lock_.test_and_set(std::memory_order_acquire)) {}
    Entry* out = nullptr;
    std::uint32_t slot = static_cast<std::uint32_t>(h) & mask_;
    for (;; slot = (slot + 1) & mask_) {
        Entry* e = table_[slot].load(std::memory_order_relaxed);
        if (e == nullptr) break;
        if (e->hash == h && e->key_len == key.size() && std::memcmp(e->key, key.data(), key.size()) == 0) { out = e; break; }
    }
    const bool full = size_.load(std::memory_order_relaxed) >= capacity_;
    if (out == nullptr && full) out = overflow_.load(std::memory_order_relaxed);
    if (out == nullptr) {
        const std::string_view label = full ? std::string_view("other") : key;
        auto* e = new (std::nothrow) Entry{h, static_cast<std::uint32_t>(key.size()), nullptr, nullptr};
        if (e == nullptr) std::abort();
        e->key = static_cast<char*>(std::malloc(key.size() + 1));
        e->slot = std::calloc(1, slot_bytes_);
        if (e->key == nullptr || e->slot == nullptr) std::abort();
        std::memcpy(e->key, key.data(), key.size());
        build_(label, e->slot, ctx_);
        out = e;
        if (full) overflow_.store(e, std::memory_order_release);
        else { table_[slot].store(e, std::memory_order_release); size_.fetch_add(1, std::memory_order_acq_rel); }
    }
    lock_.clear(std::memory_order_release);
    return out;
}

}  // namespace bolt::metrics

#endif  // BOLT_OBS_METRICS
