// log.cpp — per-thread rings, the writer thread, level table and file rotation.
#include "log_internal.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "bolt/obs/crash.h"
#include "bolt/obs/obs_limits.h"

namespace bolt::log {

const char* level_name(Level level) noexcept {
    static constexpr const char* kNames[] = {"trace", "debug", "info", "warn", "error", "fatal", "off"};
    const auto i = static_cast<std::uint32_t>(level);
    return i < 7 ? kNames[i] : "?";
}

bool parse_level(std::string_view t, Level* out) noexcept {
    assert(out != nullptr);
    for (std::uint32_t i = 0; i < 7; ++i) {
        if (t == level_name(static_cast<Level>(i))) { *out = static_cast<Level>(i); return true; }
    }
    if (t == "warning") { *out = Level::Warn; return true; }
    return false;
}

}  // namespace bolt::log

#if BOLT_OBS_LOG

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define BOLT_LOG_WRITE(fd, p, n) ::_write((fd), (p), static_cast<unsigned>(n))
#define BOLT_LOG_CLOSE ::_close
#define BOLT_LOG_OPEN(path) ::_open((path), _O_CREAT | _O_APPEND | _O_WRONLY | _O_BINARY, 0640)
#else
#include <unistd.h>
#define BOLT_LOG_WRITE(fd, p, n) ::write((fd), (p), (n))
#define BOLT_LOG_CLOSE ::close
#define BOLT_LOG_OPEN(path) ::open((path), O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC, 0640)
#endif
#if defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#endif

BOLT_LOG_MODULE("log");

namespace bolt::log {

namespace {

enum RingState : std::uint32_t { kFree = 0, kOwned = 1, kOrphan = 2 };

struct Ring {
    alignas(64) std::atomic<std::uint64_t> head{0};
    std::uint64_t cached_tail = 0;
    alignas(64) std::atomic<std::uint64_t> tail{0};
    alignas(64) std::atomic<std::uint32_t> state{kFree};
    std::uint32_t mask = 0;
    Record* slots = nullptr;
};

struct ModuleTable {
    Module modules[kMaxModules];
    std::atomic<std::uint32_t> count;
    std::atomic_flag lock;
};
constinit ModuleTable g_modules{};
constinit Module g_overflow_module{};

struct Writer {
    std::mutex tail_mu;
    std::string tail;
    std::size_t tail_cap = 0;
    std::string path;
    std::string service = "bolt";
    std::string spec = "info";
    std::thread thread;
    int fd = 2;
    std::uint64_t file_bytes = 0;
    std::uint64_t reported_dropped = 0;
    std::vector<char> buf;
    std::size_t buf_n = 0;
    std::vector<Ring*> live;
};

struct Shared {
    alignas(64) std::atomic<bool> running{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> reopen{false};
    std::atomic<std::uint8_t> format{0};
    std::atomic<std::uint8_t> default_level{static_cast<std::uint8_t>(Level::Info)};
    std::atomic<bool> pass_lock{false};
    std::atomic<std::uint64_t> passes{0};
    alignas(64) std::atomic<std::uint64_t> dropped{0};
    std::atomic<std::uint64_t> truncated{0};
    std::atomic<std::uint64_t> records[7]{};
    std::atomic<std::uint64_t> write_errors{0};
    std::atomic<std::uint64_t> rotations{0};
    std::atomic<std::uint32_t> ring_hwm{0};
    Ring* rings = nullptr;
    std::atomic<std::uint32_t> ring_cap{0};
};
constinit Shared g;
Writer* g_writer = nullptr;
std::once_flag g_rings_once;

void init_rings() noexcept {
    const auto cap = static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_log_rings));
    g.rings = new (std::nothrow) Ring[cap];
    if (g.rings != nullptr) g.ring_cap.store(cap, std::memory_order_release);
}

struct ThreadRing {
    Ring* ring = nullptr;
    ~ThreadRing();
};
constinit thread_local bool t_ring_dead = false;
thread_local ThreadRing t_ring;

ThreadRing::~ThreadRing() {
    t_ring_dead = true;
    if (ring != nullptr) {
        ring->state.store(kOrphan, std::memory_order_release);
        crash::thread_fini();
    }
}

std::uint32_t thread_id() noexcept {
    static constinit thread_local std::uint32_t t_id = 0;
    if (t_id != 0) return t_id;
#if defined(__APPLE__)
    std::uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    t_id = static_cast<std::uint32_t>(id);
#elif defined(__linux__)
    t_id = static_cast<std::uint32_t>(::syscall(SYS_gettid));
#else
    t_id = static_cast<std::uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
    return t_id;
}

std::uint32_t pow2_slots() noexcept {
    auto n = static_cast<std::uint32_t>(obs::obs_limits_value(obs::obs_limits_id::obs_log_ring_slots));
    std::uint32_t p = 16;
    while (p < n) p <<= 1;
    return p;
}

// A ring is claimed from the free pool, or reused from a thread that exited
// once the writer has emptied it.
Ring* attach_ring() noexcept {
    std::call_once(g_rings_once, init_rings);
    const std::uint32_t cap = g.ring_cap.load(std::memory_order_acquire);
    assert(cap == 0 || g.rings != nullptr);
    for (std::uint32_t i = 0; i < cap; ++i) {
        Ring& r = g.rings[i];
        std::uint32_t st = r.state.load(std::memory_order_acquire);
        const bool drained = r.head.load(std::memory_order_acquire) == r.tail.load(std::memory_order_acquire);
        if (st == kOrphan && drained &&
            r.state.compare_exchange_strong(st, kOwned, std::memory_order_acq_rel)) {
            return &r;
        }
        if (st != kFree) continue;
        if (!r.state.compare_exchange_strong(st, kOwned, std::memory_order_acq_rel)) continue;
        const std::uint32_t n = pow2_slots();
        if (r.slots == nullptr) r.slots = static_cast<Record*>(std::malloc(sizeof(Record) * n));
        if (r.slots == nullptr) { r.state.store(kFree, std::memory_order_release); return nullptr; }
        r.mask = n - 1;
        std::uint32_t hwm = g.ring_hwm.load(std::memory_order_relaxed);
        while (hwm < i + 1 && !g.ring_hwm.compare_exchange_weak(hwm, i + 1, std::memory_order_acq_rel)) {}
        return &r;
    }
    return nullptr;
}

void direct_write(const Record& r) noexcept {
    char line[detail::kFormatBytes + 512];
    const Module* m = module_at(r.module);
    const std::size_t n = detail::format_line(r, static_cast<Format>(g.format.load(std::memory_order_relaxed)),
                                              m != nullptr ? m->name : "?", "bolt", line, sizeof(line));
    if (BOLT_LOG_WRITE(2, line, n) < 0) g.write_errors.fetch_add(1, std::memory_order_relaxed);
    g.records[r.level < 7 ? r.level : 6].fetch_add(1, std::memory_order_relaxed);
}

void write_all(Writer& w, const char* p, std::size_t n) noexcept {
    while (n > 0) {
        const auto k = BOLT_LOG_WRITE(w.fd, p, n);
        if (k <= 0) { g.write_errors.fetch_add(1, std::memory_order_relaxed); return; }
        p += k;
        n -= static_cast<std::size_t>(k);
    }
}

void open_file(Writer& w) noexcept {
    assert(!w.path.empty());
    const int fd = BOLT_LOG_OPEN(w.path.c_str());
    if (fd < 0) { g.write_errors.fetch_add(1, std::memory_order_relaxed); w.fd = 2; return; }
    w.fd = fd;
    FILE* f = std::fopen(w.path.c_str(), "rb");
    w.file_bytes = 0;
    if (f != nullptr) {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        w.file_bytes = sz > 0 ? static_cast<std::uint64_t>(sz) : 0;
        std::fclose(f);
    }
}

void rotate(Writer& w) noexcept {
    assert(!w.path.empty());
    if (w.fd != 2) BOLT_LOG_CLOSE(w.fd);
    const auto keep = obs::obs_limits_value(obs::obs_limits_id::obs_log_file_keep);
    char from[1200], to[1200];
    for (std::uint64_t i = keep; i >= 1; --i) {
        std::snprintf(to, sizeof(to), "%s.%llu", w.path.c_str(), static_cast<unsigned long long>(i));
        if (i == 1) std::snprintf(from, sizeof(from), "%s", w.path.c_str());
        else std::snprintf(from, sizeof(from), "%s.%llu", w.path.c_str(), static_cast<unsigned long long>(i - 1));
        (void)std::rename(from, to);
    }
    open_file(w);
    g.rotations.fetch_add(1, std::memory_order_relaxed);
}

void flush_buf(Writer& w) noexcept {
    if (w.buf_n == 0) return;
    write_all(w, w.buf.data(), w.buf_n);
    {
        std::lock_guard<std::mutex> lock(w.tail_mu);
        w.tail.append(w.buf.data(), w.buf_n);
        if (w.tail.size() > w.tail_cap) w.tail.erase(0, w.tail.size() - w.tail_cap / 2);
    }
    w.file_bytes += w.buf_n;
    w.buf_n = 0;
    if (!w.path.empty() && w.file_bytes >= obs::obs_limits_value(obs::obs_limits_id::obs_log_file_bytes)) rotate(w);
}

void write_record(Writer& w, const Record& r) noexcept {
    if (w.buf.size() - w.buf_n < detail::kFormatBytes + 512) flush_buf(w);
    const Module* m = module_at(r.module);
    w.buf_n += detail::format_line(r, static_cast<Format>(g.format.load(std::memory_order_relaxed)),
                                   m != nullptr ? m->name : "?", w.service.c_str(),
                                   w.buf.data() + w.buf_n, w.buf.size() - w.buf_n);
    g.records[r.level < 7 ? r.level : 6].fetch_add(1, std::memory_order_relaxed);
    if ((r.flags & kFlagTruncated) != 0) g.truncated.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t ring_dropped_total() noexcept { return g.dropped.load(std::memory_order_relaxed); }

// One bounded pass: merge the ring heads in time order.
std::uint32_t drain_pass(Writer& w) noexcept {
    constexpr std::uint32_t kPassRecords = 1u << 16;
    std::vector<Ring*>& live = w.live;   // reserved in start(): no allocation here, crash_flush runs this
    live.clear();
    const std::uint32_t hwm = g.ring_hwm.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < hwm; ++i) {
        Ring& r = g.rings[i];
        if (r.tail.load(std::memory_order_relaxed) != r.head.load(std::memory_order_acquire)) live.push_back(&r);
    }
    std::uint32_t done = 0;
    while (!live.empty() && done < kPassRecords) {
        Ring* best = nullptr;
        std::uint64_t best_ts = UINT64_MAX;
        for (std::size_t i = 0; i < live.size();) {
            Ring* r = live[i];
            const std::uint64_t t = r->tail.load(std::memory_order_relaxed);
            if (t == r->head.load(std::memory_order_acquire)) { live[i] = live.back(); live.pop_back(); continue; }
            const std::uint64_t ts = r->slots[t & r->mask].ts_ns;
            if (ts < best_ts) { best_ts = ts; best = r; }
            ++i;
        }
        if (best == nullptr) break;
        const std::uint64_t t = best->tail.load(std::memory_order_relaxed);
        write_record(w, best->slots[t & best->mask]);
        best->tail.store(t + 1, std::memory_order_release);
        ++done;
    }
    return done;
}

void note_drops(Writer& w) noexcept {
    const std::uint64_t d = ring_dropped_total();
    if (d == w.reported_dropped) return;
    const std::uint64_t lost = d - w.reported_dropped;
    w.reported_dropped = d;
    BLOG_WARN("dropped %llu log records: ring full", static_cast<unsigned long long>(lost));
}

bool try_pass_lock() noexcept {
    bool expected = false;
    return g.pass_lock.compare_exchange_strong(expected, true, std::memory_order_acquire);
}

void writer_main(Writer* w) noexcept {
    assert(w != nullptr);
    const auto sleep_ms = [] { return obs::obs_limits_value(obs::obs_limits_id::obs_log_flush_ms); };
    while (!g.stop.load(std::memory_order_acquire)) {
        std::uint32_t n = 0;
        if (try_pass_lock()) {
            if (g.reopen.exchange(false, std::memory_order_acq_rel) && !w->path.empty()) {
                if (w->fd != 2) BOLT_LOG_CLOSE(w->fd);
                open_file(*w);
            }
            n = drain_pass(*w);
            note_drops(*w);
            flush_buf(*w);
            g.pass_lock.store(false, std::memory_order_release);
            g.passes.fetch_add(1, std::memory_order_acq_rel);
        }
        if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms()));
    }
    while (!try_pass_lock()) std::this_thread::yield();
    while (drain_pass(*w) != 0) {}
    note_drops(*w);
    flush_buf(*w);
    g.pass_lock.store(false, std::memory_order_release);
}

void apply_spec_to(Module* m, std::string_view spec) noexcept {
    std::uint8_t level = g.default_level.load(std::memory_order_relaxed);
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = std::min(spec.find(',', pos), spec.size());
        const std::string_view item = spec.substr(pos, comma - pos);
        const std::size_t eq = item.find('=');
        Level l;
        if (eq != std::string_view::npos && item.substr(0, eq) == m->name && parse_level(item.substr(eq + 1), &l)) {
            level = static_cast<std::uint8_t>(l);
        }
        pos = comma + 1;
    }
    m->level.store(level, std::memory_order_relaxed);
}

std::mutex& spec_mu() { static std::mutex mu; return mu; }
std::string& spec_text() { static std::string s = "info"; return s; }

}  // namespace

Module* intern(const char* name) noexcept {
    assert(name != nullptr);
    while (g_modules.lock.test_and_set(std::memory_order_acquire)) {}
    const std::uint32_t n = g_modules.count.load(std::memory_order_relaxed);
    Module* out = nullptr;
    for (std::uint32_t i = 0; i < n && out == nullptr; ++i) {
        if (std::strncmp(g_modules.modules[i].name, name, kModuleNameBytes - 1) == 0) out = &g_modules.modules[i];
    }
    if (out == nullptr && n < kMaxModules) {
        out = &g_modules.modules[n];
        out->id = static_cast<std::uint16_t>(n);
        std::strncpy(out->name, name, kModuleNameBytes - 1);
        out->level.store(g.default_level.load(std::memory_order_relaxed), std::memory_order_relaxed);
        g_modules.count.store(n + 1, std::memory_order_release);
    }
    g_modules.lock.clear(std::memory_order_release);
    if (out != nullptr) return out;
    g_overflow_module.level.store(static_cast<std::uint8_t>(Level::Info), std::memory_order_relaxed);
    std::strncpy(g_overflow_module.name, "other", kModuleNameBytes - 1);
    return &g_overflow_module;
}

std::uint32_t module_count() noexcept { return g_modules.count.load(std::memory_order_acquire); }
const Module* module_at(std::uint32_t i) noexcept { return i < module_count() ? &g_modules.modules[i] : nullptr; }

bool set_spec(std::string_view spec, char* err, std::size_t err_cap) noexcept {
    std::uint8_t def = g.default_level.load(std::memory_order_relaxed);
    std::size_t pos = 0;
    while (pos <= spec.size() && !spec.empty()) {
        const std::size_t comma = std::min(spec.find(',', pos), spec.size());
        const std::string_view item = spec.substr(pos, comma - pos);
        const std::size_t eq = item.find('=');
        const std::string_view text = eq == std::string_view::npos ? item : item.substr(eq + 1);
        Level l;
        if (!parse_level(text, &l)) {
            if (err != nullptr) std::snprintf(err, err_cap, "bad log level '%.*s'", static_cast<int>(text.size()), text.data());
            return false;
        }
        if (eq == std::string_view::npos) def = static_cast<std::uint8_t>(l);
        pos = comma + 1;
    }
    std::lock_guard<std::mutex> lock(spec_mu());
    g.default_level.store(def, std::memory_order_relaxed);
    spec_text().assign(spec.data(), spec.size());
    const std::uint32_t n = module_count();
    for (std::uint32_t i = 0; i < n; ++i) apply_spec_to(&g_modules.modules[i], spec);
    return true;
}

std::string get_spec() {
    std::lock_guard<std::mutex> lock(spec_mu());
    return spec_text();
}

void set_format(Format f) noexcept { g.format.store(static_cast<std::uint8_t>(f), std::memory_order_relaxed); }
Format format() noexcept { return static_cast<Format>(g.format.load(std::memory_order_relaxed)); }

bool start(const Config& c) noexcept {
    assert(c.service != nullptr);
    if (g.running.load(std::memory_order_acquire)) return true;
    set_format(c.format);
    if (c.spec != nullptr && !set_spec(c.spec, nullptr, 0)) return false;
    std::call_once(g_rings_once, init_rings);
    if (g.ring_cap.load(std::memory_order_acquire) == 0) return false;
    auto* w = new (std::nothrow) Writer();
    if (w == nullptr) return false;
    w->service = c.service;
    w->tail_cap = obs::obs_limits_value(obs::obs_limits_id::obs_log_tail_bytes);
    w->buf.resize(1u << 16);
    w->live.reserve(g.ring_cap.load(std::memory_order_acquire));
    if (c.path != nullptr && c.path[0] != '\0') {
        w->path = c.path;
        open_file(*w);
    }
    g_writer = w;
    g.stop.store(false, std::memory_order_release);
    g.running.store(true, std::memory_order_release);
    w->thread = std::thread(writer_main, w);
    return true;
}

void stop() noexcept {
    if (!g.running.exchange(false, std::memory_order_acq_rel)) return;
    g.stop.store(true, std::memory_order_release);
    Writer* w = g_writer;
    assert(w != nullptr);
    w->thread.join();
    if (w->fd != 2) { BOLT_LOG_CLOSE(w->fd); w->fd = 2; }
}

bool running() noexcept { return g.running.load(std::memory_order_acquire); }

void flush() noexcept {
    if (!running()) return;
    const std::uint64_t seen = g.passes.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < 2000 && g.passes.load(std::memory_order_acquire) < seen + 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void reopen() noexcept { g.reopen.store(true, std::memory_order_release); }

Stats stats() noexcept {
    Stats s{};
    for (std::uint32_t i = 0; i < 7; ++i) s.records[i] = g.records[i].load(std::memory_order_relaxed);
    s.dropped = g.dropped.load(std::memory_order_relaxed);
    s.truncated = g.truncated.load(std::memory_order_relaxed);
    s.write_errors = g.write_errors.load(std::memory_order_relaxed);
    s.rotations = g.rotations.load(std::memory_order_relaxed);
    s.rings_capacity = g.ring_cap.load(std::memory_order_relaxed);
    const std::uint32_t hwm = g.ring_hwm.load(std::memory_order_acquire);
    for (std::uint32_t i = 0; i < hwm; ++i) {
        if (g.rings[i].state.load(std::memory_order_relaxed) == kOwned) ++s.rings_in_use;
    }
    return s;
}

void tail(std::string* out, std::size_t max_bytes) {
    assert(out != nullptr);
    Writer* w = g_writer;
    if (w == nullptr) return;
    std::lock_guard<std::mutex> lock(w->tail_mu);
    const std::size_t n = std::min(max_bytes, w->tail.size());
    out->assign(w->tail, w->tail.size() - n, n);
}

void crash_flush(int fd) noexcept {
    Writer* w = g_writer;
    if (w == nullptr) return;
    for (std::uint32_t i = 0; i < 100000 && !try_pass_lock(); ++i) {}
    (void)drain_pass(*w);
    if (w->tail_mu.try_lock()) {
        if (fd >= 0) (void)!BOLT_LOG_WRITE(fd, w->tail.data(), w->tail.size());
        w->tail_mu.unlock();
    }
    if (fd >= 0 && w->buf_n != 0) (void)!BOLT_LOG_WRITE(fd, w->buf.data(), w->buf_n);
    if (w->buf_n != 0) write_all(*w, w->buf.data(), w->buf_n);
    w->buf_n = 0;
}

namespace detail {

Record* begin() noexcept {
    constinit static thread_local Record t_scratch{};
    if (!g.running.load(std::memory_order_acquire) || t_ring_dead) return &t_scratch;
    ThreadRing& tr = t_ring;
    if (tr.ring == nullptr) {
        tr.ring = attach_ring();
        if (tr.ring == nullptr) { g.dropped.fetch_add(1, std::memory_order_relaxed); return nullptr; }
        crash::thread_init();
    }
    Ring& r = *tr.ring;
    const std::uint64_t h = r.head.load(std::memory_order_relaxed);
    if (h - r.cached_tail > r.mask) {
        r.cached_tail = r.tail.load(std::memory_order_acquire);
        if (h - r.cached_tail > r.mask) { g.dropped.fetch_add(1, std::memory_order_relaxed); return nullptr; }
    }
    return &r.slots[h & r.mask];
}

void commit(Record* record) noexcept {
    assert(record != nullptr);
    Ring* r = t_ring_dead ? nullptr : t_ring.ring;
    if (r == nullptr || record < r->slots || record > r->slots + r->mask) { direct_write(*record); return; }
    r->head.store(r->head.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

void stamp(Record* r, const Module* m, Level l, const char* file, std::uint32_t line, const char* fmt) noexcept {
    assert(r != nullptr && m != nullptr);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    r->ts_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    const obs::TraceCtx& c = obs::trace_ctx();
    r->trace_hi = c.hi;
    r->trace_lo = c.lo;
    r->span = c.span;
    r->fmt = fmt;
    r->file = file;
    r->line = line;
    r->tid = thread_id();
    r->module = m->id;
    r->level = static_cast<std::uint8_t>(l);
    r->flags = 0;
    r->payload_len = 0;
}

bool arg_has_precision(const char* f, std::uint32_t arg) noexcept {
    assert(f != nullptr);
    std::uint32_t used = 0;
    while (*f != '\0') {
        if (*f++ != '%') continue;
        if (*f == '%') { ++f; continue; }
        bool prec = false;
        for (; *f != '\0' && std::strchr("-+ #0123456789.*hlqjztL", *f) != nullptr; ++f) {
            if (*f == '*') { ++used; if (f[-1] == '.') prec = true; }
        }
        if (*f == '\0') return false;
        ++f;
        if (used == arg) return prec;
        ++used;
    }
    return false;
}

void put_str(Enc& e, const char* s, std::size_t n) noexcept {
    const std::size_t room = static_cast<std::size_t>(e.end - e.p);
    if (room < 3) { e.truncated = true; return; }
    if (n > room - 3) { n = room - 3; e.truncated = true; }
    *e.p++ = kArgStr;
    const auto n16 = static_cast<std::uint16_t>(n);
    std::memcpy(e.p, &n16, 2);
    e.p += 2;
    if (n != 0) std::memcpy(e.p, s, n);
    e.p += n;
}

void put_cstr(Enc& e, const char* s) noexcept {
    if (s == nullptr) { put_str(e, "(null)", 6); return; }
    const std::size_t room = static_cast<std::size_t>(e.end - e.p);
    std::size_t cap = room + 1;
    if (e.prev_was_int && arg_has_precision(e.fmt, e.arg)) cap = static_cast<std::size_t>(e.prev_int);
    std::size_t n = 0;
    while (n < cap && s[n] != '\0') ++n;
    put_str(e, s, n);
}

void emit_event(const Module* m, Level l, const char* file, std::uint32_t line,
                const char* event, std::string_view fields) noexcept {
    Record* r = begin();
    if (r == nullptr) return;
    stamp(r, m, l, file, line, event);
    r->flags = kFlagEvent;
    if (fields.size() <= kPayloadBytes) {
        std::memcpy(r->payload, fields.data(), fields.size());
        r->payload_len = static_cast<std::uint16_t>(fields.size());
    } else {
        constexpr std::string_view kCut = "\"fields_truncated\":true";
        std::memcpy(r->payload, kCut.data(), kCut.size());
        r->payload_len = static_cast<std::uint16_t>(kCut.size());
        r->flags |= kFlagTruncated;
    }
    commit(r);
}

}  // namespace detail

}  // namespace bolt::log

#endif  // BOLT_OBS_LOG
