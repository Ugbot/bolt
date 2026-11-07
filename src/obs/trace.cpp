// trace.cpp — sampling, the bounded span queue and its drain thread.
#include "bolt/obs/trace.h"

#if BOLT_OBS_TRACE

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "bolt/obs/obs_limits.h"

namespace bolt::trace {

namespace {

struct Slot {
    std::atomic<std::uint64_t> seq;
    SpanRecord rec;
};

struct State {
    alignas(64) std::atomic<std::uint32_t> ppm{0};
    alignas(64) std::atomic<std::uint64_t> enqueue{0};
    alignas(64) std::atomic<std::uint64_t> dequeue{0};
    std::atomic<std::uint64_t> finished{0}, dropped{0}, exported{0}, export_errors{0};
    std::atomic<bool> running{false};
    Slot* slots = nullptr;
    std::uint64_t mask = 0;
    Sink sink = nullptr;
    void* sink_ctx = nullptr;
    std::thread thread;
    std::mutex retained_mu;
    std::vector<SpanRecord> retained;
    std::uint64_t retained_next = 0;
};
State& state() { static State* s = new State(); return *s; }

std::uint64_t unix_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::uint64_t pow2_at_least(std::uint64_t n) noexcept {
    std::uint64_t p = 64;
    while (p < n) p <<= 1;
    return p;
}

// Vyukov bounded MPMC push; drop on full.
bool push(State& s, const SpanRecord& r) noexcept {
    std::uint64_t pos = s.enqueue.load(std::memory_order_relaxed);
    for (std::uint32_t tries = 0; tries < 64; ++tries) {
        Slot& slot = s.slots[pos & s.mask];
        const std::uint64_t seq = slot.seq.load(std::memory_order_acquire);
        const auto dif = static_cast<std::int64_t>(seq) - static_cast<std::int64_t>(pos);
        if (dif == 0) {
            if (s.enqueue.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                slot.rec = r;
                slot.seq.store(pos + 1, std::memory_order_release);
                return true;
            }
        } else if (dif < 0) {
            return false;
        } else {
            pos = s.enqueue.load(std::memory_order_relaxed);
        }
    }
    return false;
}

std::uint32_t pop_batch(State& s, SpanRecord* out, std::uint32_t cap) noexcept {
    std::uint32_t n = 0;
    std::uint64_t pos = s.dequeue.load(std::memory_order_relaxed);
    while (n < cap) {
        Slot& slot = s.slots[pos & s.mask];
        if (slot.seq.load(std::memory_order_acquire) != pos + 1) break;
        out[n++] = slot.rec;
        slot.seq.store(pos + s.mask + 1, std::memory_order_release);
        ++pos;
    }
    s.dequeue.store(pos, std::memory_order_relaxed);
    return n;
}

void retain(State& s, const SpanRecord* spans, std::uint32_t n) {
    std::lock_guard<std::mutex> lock(s.retained_mu);
    for (std::uint32_t i = 0; i < n; ++i) {
        s.retained[s.retained_next % s.retained.size()] = spans[i];
        ++s.retained_next;
    }
}

void drain_main(State* s) {
    assert(s != nullptr);
    constexpr std::uint32_t kBatch = 512;
    std::vector<SpanRecord> batch(kBatch);
    for (;;) {
        const bool run = s->running.load(std::memory_order_acquire);
        const std::uint32_t n = pop_batch(*s, batch.data(), kBatch);
        if (n != 0) {
            retain(*s, batch.data(), n);
            if (s->sink != nullptr) s->sink(s->sink_ctx, batch.data(), n);
        }
        if (n == 0 && !run) return;
        if (n < kBatch) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

}  // namespace

void set_rate_ppm(std::uint32_t ppm) noexcept {
    state().ppm.store(ppm > 1000000u ? 1000000u : ppm, std::memory_order_relaxed);
}
std::uint32_t rate_ppm() noexcept { return state().ppm.load(std::memory_order_relaxed); }

bool sample(std::uint64_t lo) noexcept {
    static State& s = state();
    const std::uint32_t ppm = s.ppm.load(std::memory_order_relaxed);
    if (ppm == 0 || !s.running.load(std::memory_order_relaxed)) return false;
    return (lo * 0x9e3779b97f4a7c15ull >> 32) % 1000000u < ppm;
}

std::uint64_t new_span_id() noexcept {
    static constinit thread_local std::uint64_t t_state = 0;
    if (t_state == 0) t_state = unix_ns() ^ reinterpret_cast<std::uintptr_t>(&t_state);
    t_state ^= t_state << 13; t_state ^= t_state >> 7; t_state ^= t_state << 17;
    return t_state != 0 ? t_state : 1;
}

bool start(Sink sink, void* ctx) noexcept {
    State& s = state();
    if (s.running.load(std::memory_order_acquire)) return true;
    const std::uint64_t n = pow2_at_least(obs::obs_limits_value(obs::obs_limits_id::obs_trace_ring_spans));
    s.slots = new (std::nothrow) Slot[n];
    if (s.slots == nullptr) return false;
    for (std::uint64_t i = 0; i < n; ++i) s.slots[i].seq.store(i, std::memory_order_relaxed);
    s.mask = n - 1;
    s.retained.resize(obs::obs_limits_value(obs::obs_limits_id::obs_trace_retained_spans));
    s.sink = sink;
    s.sink_ctx = ctx;
    s.running.store(true, std::memory_order_release);
    s.thread = std::thread(drain_main, &s);
    set_rate_ppm(static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_trace_sample_ppm)));
    return true;
}

void stop() noexcept {
    State& s = state();
    if (!s.running.exchange(false, std::memory_order_acq_rel)) return;
    s.thread.join();
}

void note_export(bool ok, std::uint32_t n) noexcept {
    (ok ? state().exported : state().export_errors).fetch_add(ok ? n : 1, std::memory_order_relaxed);
}

Stats stats() noexcept {
    State& s = state();
    return Stats{s.finished.load(std::memory_order_relaxed), s.dropped.load(std::memory_order_relaxed),
                 s.exported.load(std::memory_order_relaxed), s.export_errors.load(std::memory_order_relaxed)};
}

void recent_json(std::string* out, std::uint32_t limit, std::uint64_t trace_lo) {
    assert(out != nullptr);
    State& s = state();
    std::lock_guard<std::mutex> lock(s.retained_mu);
    out->push_back('[');
    if (!s.retained.empty()) {
        const std::uint64_t size = s.retained.size();
        const std::uint64_t have = s.retained_next < size ? s.retained_next : size;
        std::uint32_t emitted = 0;
        for (std::uint64_t i = 0; i < have && emitted < limit; ++i) {
            const SpanRecord& r = s.retained[(s.retained_next - 1 - i) % size];
            if (trace_lo != 0 && r.trace_lo != trace_lo) continue;
            char b[320];
            std::snprintf(b, sizeof(b),
                "%s{\"trace_id\":\"%016llx%016llx\",\"span_id\":\"%016llx\",\"parent_id\":\"%016llx\",\"name\":\"%s\","
                "\"start_ns\":\"%llu\",\"duration_ns\":\"%llu\",\"error\":%s",
                emitted != 0 ? "," : "", (unsigned long long)r.trace_hi, (unsigned long long)r.trace_lo,
                (unsigned long long)r.span, (unsigned long long)r.parent, r.name, (unsigned long long)r.start_ns,
                (unsigned long long)(r.end_ns - r.start_ns), r.error != 0 ? "true" : "false");
            out->append(b);
            if (r.attrs[0] != '\0') out->append(",").append(r.attrs);
            out->push_back('}');
            ++emitted;
        }
    }
    out->push_back(']');
}

void Span::begin(const char* name) noexcept {
    assert(name != nullptr);
    obs::TraceCtx& c = obs::t_trace_ctx;
    rec_.trace_hi = c.hi;
    rec_.trace_lo = c.lo;
    rec_.parent = c.span;
    rec_.span = new_span_id();
    rec_.name = name;
    rec_.start_ns = unix_ns();
    rec_.attrs[0] = '\0';
    rec_.error = 0;
    saved_parent_ = c.span;
    c.span = rec_.span;
    active_ = true;
}

void Span::end() noexcept {
    assert(active_);
    obs::t_trace_ctx.span = saved_parent_;
    rec_.end_ns = unix_ns();
    State& s = state();
    s.finished.fetch_add(1, std::memory_order_relaxed);
    if (s.slots == nullptr || !push(s, rec_)) s.dropped.fetch_add(1, std::memory_order_relaxed);
}

void Span::attrs(std::string_view f) noexcept {
    if (!active_) return;
    const std::size_t n = f.size() < kAttrBytes - 1 ? f.size() : 0;   // a cut fragment would not be JSON
    std::memcpy(rec_.attrs, f.data(), n);
    rec_.attrs[n] = '\0';
}

}  // namespace bolt::trace

#endif  // BOLT_OBS_TRACE
