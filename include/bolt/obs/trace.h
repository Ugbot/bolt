// bolt/obs/trace.h — sampled spans. An unsampled request pays one thread-local
// load per span site. Finished spans go through a bounded queue to a background
// thread, which retains them and hands batches to a sink (OTLP lives in the host).
// With BOLT_OBS_TRACE=0 a Span is empty and sampling is constantly false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "bolt/obs/context.h"

namespace bolt::trace {

inline constexpr std::uint32_t kAttrBytes = 104;

struct SpanRecord {
    std::uint64_t trace_hi, trace_lo;
    std::uint64_t span, parent;
    std::uint64_t start_ns, end_ns;   // unix nanoseconds
    const char*   name;               // a literal
    char          attrs[kAttrBytes];  // JSON fields, may be empty
    std::uint8_t  error;
};

using Sink = void (*)(void* ctx, const SpanRecord* spans, std::uint32_t n);

struct Stats {
    std::uint64_t finished;
    std::uint64_t dropped;
    std::uint64_t exported;
    std::uint64_t export_errors;
};

#if BOLT_OBS_TRACE

void set_rate_ppm(std::uint32_t ppm) noexcept;
std::uint32_t rate_ppm() noexcept;
bool sample(std::uint64_t trace_lo) noexcept;   // head decision; one load when the rate is 0
std::uint64_t new_span_id() noexcept;
bool start(Sink sink, void* ctx) noexcept;      // sink may be null: retain only
void stop() noexcept;
void note_export(bool ok, std::uint32_t n) noexcept;
Stats stats() noexcept;
void recent_json(std::string* out, std::uint32_t limit, std::uint64_t trace_lo);

class Span {
public:
    explicit Span(const char* name) noexcept {
        if (obs::t_trace_ctx.sampled != 0) begin(name);
    }
    ~Span() { if (active_) end(); }
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
    bool active() const noexcept { return active_; }
    void set_error() noexcept { if (active_) rec_.error = 1; }
    void attrs(std::string_view json_fields) noexcept;
private:
    void begin(const char* name) noexcept;
    void end() noexcept;
    bool active_ = false;
    std::uint64_t saved_parent_ = 0;
    SpanRecord rec_;
};

#else

inline void set_rate_ppm(std::uint32_t) noexcept {}
inline std::uint32_t rate_ppm() noexcept { return 0; }
inline constexpr bool sample(std::uint64_t) noexcept { return false; }
inline std::uint64_t new_span_id() noexcept { return 0; }
inline bool start(Sink, void*) noexcept { return true; }
inline void stop() noexcept {}
inline void note_export(bool, std::uint32_t) noexcept {}
inline Stats stats() noexcept { return Stats{}; }
inline void recent_json(std::string*, std::uint32_t, std::uint64_t) {}

class Span {
public:
    explicit Span(const char*) noexcept {}
    bool active() const noexcept { return false; }
    void set_error() noexcept {}
    void attrs(std::string_view) noexcept {}
};

#endif

}  // namespace bolt::trace
