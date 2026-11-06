// bolt/obs/log.h — structured logging that never blocks its caller.
//
// A call site below the module's level costs one relaxed load. Above it, the
// format pointer and the raw arguments are copied into a fixed record in the
// calling thread's SPSC ring; a background thread formats and writes. A full
// ring drops the record and counts it. Before start() (and after stop()) a
// record is formatted and written synchronously, so tools and tests need no
// setup. With BOLT_OBS_LOG=0, or below BOLT_OBS_LOG_MIN, a call site is nothing.
//
//   BOLT_LOG_MODULE("tiering");            // once per .cpp
//   BLOG_WARN("upload failed: %s (%d)", why, code);
//
// fmt must be a string literal: only its pointer is stored.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

#include "bolt/obs/context.h"

#if defined(_MSC_VER)
#define BOLT_LOG_COLD __declspec(noinline)
#else
#define BOLT_LOG_COLD __attribute__((noinline, cold))
#endif

namespace bolt::log {

enum class Level : std::uint8_t { Trace = 0, Debug, Info, Warn, Error, Fatal, Off };
enum class Format : std::uint8_t { Text = 0, Json };

inline constexpr std::uint32_t kModuleNameBytes = 24;
inline constexpr std::uint32_t kMaxModules = 256;

struct Module {
    std::atomic<std::uint8_t> level;
    std::uint16_t id;
    char name[kModuleNameBytes];
};

struct Config {
    Format format = Format::Text;
    const char* path = nullptr;   // nullptr: stderr
    const char* spec = nullptr;   // "info,tiering=debug"
    const char* service = "bolt";
};

struct Stats {
    std::uint64_t records[7];     // written, by level
    std::uint64_t dropped;        // ring full or no ring
    std::uint64_t truncated;      // payload past the record size
    std::uint64_t write_errors;
    std::uint64_t rotations;
    std::uint32_t rings_in_use;
    std::uint32_t rings_capacity;
};

const char* level_name(Level level) noexcept;
bool parse_level(std::string_view text, Level* out) noexcept;

#if BOLT_OBS_LOG

inline constexpr std::uint32_t kRecordBytes = 512;
inline constexpr std::uint32_t kHeaderBytes = 64;
inline constexpr std::uint32_t kPayloadBytes = kRecordBytes - kHeaderBytes;

inline constexpr std::uint8_t kFlagTruncated = 1;
inline constexpr std::uint8_t kFlagEvent = 2;   // fmt is an event name, payload is JSON fields

struct Record {
    std::uint64_t ts_ns;
    std::uint64_t trace_hi;
    std::uint64_t trace_lo;
    std::uint64_t span;
    const char*   fmt;
    const char*   file;
    std::uint32_t line;
    std::uint32_t tid;
    std::uint16_t module;
    std::uint16_t payload_len;
    std::uint8_t  level;
    std::uint8_t  flags;
    std::uint8_t  _pad[2];
    std::uint8_t  payload[kPayloadBytes];
};
static_assert(sizeof(Record) == kRecordBytes);
static_assert(offsetof(Record, payload) == kHeaderBytes);

enum ArgTag : std::uint8_t { kArgI64 = 1, kArgU64, kArgF64, kArgPtr, kArgStr };

Module* intern(const char* name) noexcept;
bool start(const Config& config) noexcept;
void stop() noexcept;
bool running() noexcept;
void flush() noexcept;           // waits until every record committed so far is written
void reopen() noexcept;          // after an external rotate
bool set_spec(std::string_view spec, char* err, std::size_t err_cap) noexcept;
std::string get_spec();
void set_format(Format format) noexcept;
Format format() noexcept;
Stats stats() noexcept;
void tail(std::string* out, std::size_t max_bytes);
void crash_flush(int fd) noexcept;   // best effort, from a signal handler
std::uint32_t module_count() noexcept;
const Module* module_at(std::uint32_t i) noexcept;

namespace detail {

struct Enc {
    std::uint8_t* p;
    std::uint8_t* end;
    const char*   fmt;
    std::uint32_t arg;      // index of the argument being encoded
    std::uint64_t prev_int; // the previous argument, when integral
    bool          prev_was_int;
    bool          truncated;
};

Record* begin() noexcept;
void    commit(Record* record) noexcept;
void    stamp(Record* r, const Module* m, Level l, const char* file, std::uint32_t line, const char* fmt) noexcept;
bool    arg_has_precision(const char* fmt, std::uint32_t arg) noexcept;
void    put_str(Enc& e, const char* s, std::size_t n) noexcept;
void    put_cstr(Enc& e, const char* s) noexcept;

inline void put_word(Enc& e, ArgTag tag, std::uint64_t bits) noexcept {
    if (e.p + 9 > e.end) { e.truncated = true; return; }
    *e.p++ = tag;
    std::memcpy(e.p, &bits, 8);
    e.p += 8;
}

template <class T>
inline void put(Enc& e, const T& v) noexcept {
    using D = std::decay_t<T>;
    bool is_int = false;
    if constexpr (std::is_same_v<D, const char*> || std::is_same_v<D, char*>) {
        put_cstr(e, v);
    } else if constexpr (std::is_same_v<D, std::string_view>) {
        put_str(e, v.data(), v.size());
    } else if constexpr (std::is_enum_v<D>) {
        put_word(e, kArgI64, static_cast<std::uint64_t>(v));
    } else if constexpr (std::is_integral_v<D>) {
        is_int = true;
        const std::uint64_t bits = static_cast<std::uint64_t>(static_cast<long long>(v));
        put_word(e, std::is_signed_v<D> ? kArgI64 : kArgU64, bits);
        e.prev_int = bits;
    } else if constexpr (std::is_floating_point_v<D>) {
        const double d = static_cast<double>(v);
        std::uint64_t bits;
        std::memcpy(&bits, &d, 8);
        put_word(e, kArgF64, bits);
    } else {
        static_assert(std::is_pointer_v<D>, "unsupported log argument");
        put_word(e, kArgPtr, reinterpret_cast<std::uintptr_t>(v));
    }
    e.prev_was_int = is_int;
    ++e.arg;
}

template <class... A>
BOLT_LOG_COLD void emit(const Module* m, Level l, const char* file, std::uint32_t line,
                                          const char* fmt, const A&... a) noexcept {
    Record* r = begin();
    if (r == nullptr) return;
    stamp(r, m, l, file, line, fmt);
    Enc e{r->payload, r->payload + kPayloadBytes, fmt, 0, 0, false, false};
    (put(e, a), ...);
    r->payload_len = static_cast<std::uint16_t>(e.p - r->payload);
    if (e.truncated) r->flags |= kFlagTruncated;
    commit(r);
}

void emit_event(const Module* m, Level l, const char* file, std::uint32_t line,
                const char* event, std::string_view json_fields) noexcept;

}  // namespace detail

inline bool enabled(const Module* m, Level l) noexcept {
    return static_cast<std::uint8_t>(l) >= m->level.load(std::memory_order_relaxed);
}

#else  // BOLT_OBS_LOG

inline bool start(const Config&) noexcept { return true; }
inline void stop() noexcept {}
inline bool running() noexcept { return false; }
inline void flush() noexcept {}
inline void reopen() noexcept {}
inline bool set_spec(std::string_view, char*, std::size_t) noexcept { return true; }
inline std::string get_spec() { return "compiled_out"; }
inline void set_format(Format) noexcept {}
inline Format format() noexcept { return Format::Text; }
inline Stats stats() noexcept { return Stats{}; }
inline void tail(std::string*, std::size_t) {}
inline void crash_flush(int) noexcept {}
inline std::uint32_t module_count() noexcept { return 0; }
inline const Module* module_at(std::uint32_t) noexcept { return nullptr; }

#endif  // BOLT_OBS_LOG

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void check_format(const char*, ...) noexcept {}

}  // namespace bolt::log

#define BOLT_LOG_LEVEL_Trace 0
#define BOLT_LOG_LEVEL_Debug 1
#define BOLT_LOG_LEVEL_Info 2
#define BOLT_LOG_LEVEL_Warn 3
#define BOLT_LOG_LEVEL_Error 4
#define BOLT_LOG_LEVEL_Fatal 5

#if BOLT_OBS_LOG
#define BOLT_LOG_MODULE(name) \
    [[maybe_unused]] static ::bolt::log::Module* const bolt_log_module_ = ::bolt::log::intern(name)
#define BOLT_LOG_AT_(lvl, fmt, ...)                                                                 \
    do {                                                                                            \
        if constexpr (BOLT_LOG_LEVEL_##lvl >= BOLT_OBS_LOG_MIN) {                                   \
            if (::bolt::log::enabled(bolt_log_module_, ::bolt::log::Level::lvl)) {                  \
                ::bolt::log::detail::emit(bolt_log_module_, ::bolt::log::Level::lvl, __FILE__,      \
                                          __LINE__, fmt __VA_OPT__(, ) __VA_ARGS__);                \
            }                                                                                       \
        }                                                                                           \
        if (false) ::bolt::log::check_format(fmt __VA_OPT__(, ) __VA_ARGS__);                       \
    } while (0)
#define BLOG_EVENT(lvl, event, json_fields)                                                         \
    do {                                                                                            \
        if (::bolt::log::enabled(bolt_log_module_, ::bolt::log::Level::lvl)) {                      \
            ::bolt::log::detail::emit_event(bolt_log_module_, ::bolt::log::Level::lvl, __FILE__,    \
                                            __LINE__, event, json_fields);                          \
        }                                                                                           \
    } while (0)
#else
#define BOLT_LOG_MODULE(name) static_assert(true)
#define BOLT_LOG_AT_(lvl, fmt, ...)                                                                 \
    do {                                                                                            \
        if (false) ::bolt::log::check_format(fmt __VA_OPT__(, ) __VA_ARGS__);                       \
    } while (0)
#define BLOG_EVENT(lvl, event, json_fields) do { if (false) { (void)(json_fields); } } while (0)
#endif

#define BLOG_TRACE(fmt, ...) BOLT_LOG_AT_(Trace, fmt __VA_OPT__(, ) __VA_ARGS__)
#define BLOG_DEBUG(fmt, ...) BOLT_LOG_AT_(Debug, fmt __VA_OPT__(, ) __VA_ARGS__)
#define BLOG_INFO(fmt, ...)  BOLT_LOG_AT_(Info, fmt __VA_OPT__(, ) __VA_ARGS__)
#define BLOG_WARN(fmt, ...)  BOLT_LOG_AT_(Warn, fmt __VA_OPT__(, ) __VA_ARGS__)
#define BLOG_ERROR(fmt, ...) BOLT_LOG_AT_(Error, fmt __VA_OPT__(, ) __VA_ARGS__)
#define BLOG_FATAL(fmt, ...) BOLT_LOG_AT_(Fatal, fmt __VA_OPT__(, ) __VA_ARGS__)
