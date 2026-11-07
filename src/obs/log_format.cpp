// log_format.cpp — render a Record: printf over captured arguments, text or JSON line.
#include "log_internal.h"

#if BOLT_OBS_LOG

#include <cassert>
#include <cstdio>
#include <ctime>

#if defined(__clang__)
#define BOLT_LOG_NONLITERAL_BEGIN _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wformat-nonliteral\"")
#define BOLT_LOG_NONLITERAL_END _Pragma("clang diagnostic pop")
#else
#define BOLT_LOG_NONLITERAL_BEGIN
#define BOLT_LOG_NONLITERAL_END
#endif

namespace bolt::log::detail {

namespace {

struct Reader {
    const std::uint8_t* p;
    const std::uint8_t* end;
};

struct Arg {
    std::uint8_t tag = 0;
    std::uint64_t bits = 0;
    const char* str = nullptr;
    std::uint32_t len = 0;
};

bool next_arg(Reader& r, Arg* a) noexcept {
    if (r.p >= r.end) return false;
    a->tag = *r.p++;
    if (a->tag == kArgStr) {
        if (r.p + 2 > r.end) return false;
        std::uint16_t n;
        std::memcpy(&n, r.p, 2);
        r.p += 2;
        if (r.p + n > r.end) return false;
        a->str = reinterpret_cast<const char*>(r.p);
        a->len = n;
        r.p += n;
        return true;
    }
    if (r.p + 8 > r.end) return false;
    std::memcpy(&a->bits, r.p, 8);
    r.p += 8;
    return true;
}

struct Out {
    char* p;
    char* end;
    void raw(const char* s, std::size_t n) noexcept {
        const std::size_t room = static_cast<std::size_t>(end - p);
        if (n > room) n = room;
        std::memcpy(p, s, n);
        p += n;
    }
    void str(const char* s) noexcept { raw(s, std::strlen(s)); }
    void ch(char c) noexcept { if (p < end) *p++ = c; }
    void advance(int n) noexcept {
        if (n <= 0) return;
        const std::size_t room = static_cast<std::size_t>(end - p);
        p += static_cast<std::size_t>(n) < room ? static_cast<std::size_t>(n) : room;
    }
};

struct Spec {
    char head[24];        // "%-10.3" without the length modifier
    std::uint32_t head_n; // chars in head
    char conv;
    bool star_width;
    bool star_prec;
    bool wide;            // length modifier asks for 64 bits
};

// Parse the conversion after '%'. Returns the char after it, or nullptr.
const char* parse_spec(const char* f, Spec* s) noexcept {
    s->head[0] = '%';
    s->head_n = 1;
    s->star_width = s->star_prec = s->wide = false;
    auto push = [s](char c) { if (s->head_n + 6 < sizeof(s->head)) s->head[s->head_n++] = c; };
    while (*f != '\0' && std::strchr("-+ #0", *f) != nullptr) push(*f++);
    if (*f == '*') { s->star_width = true; push('*'); ++f; }
    while (*f >= '0' && *f <= '9') push(*f++);
    if (*f == '.') {
        push(*f++);
        if (*f == '*') { s->star_prec = true; push('*'); ++f; }
        while (*f >= '0' && *f <= '9') push(*f++);
    }
    while (*f != '\0' && std::strchr("hlqjztL", *f) != nullptr) {
        if (*f == 'l' || *f == 'q' || *f == 'j' || *f == 'z' || *f == 't') s->wide = true;
        ++f;
    }
    if (*f == '\0') return nullptr;
    s->conv = *f;
    return f + 1;
}

void emit_int(Out& o, Spec& s, const int* stars, std::uint32_t n_stars, const Arg& a) noexcept {
    std::uint64_t v = a.bits;
    const bool is_unsigned = s.conv == 'u' || s.conv == 'x' || s.conv == 'X' || s.conv == 'o';
    if (is_unsigned && !s.wide) v &= 0xffffffffu;
    char spec[32];
    std::memcpy(spec, s.head, s.head_n);
    std::uint32_t n = s.head_n;
    spec[n++] = 'l';
    spec[n++] = 'l';
    spec[n++] = s.conv;
    spec[n] = '\0';
    const std::size_t room = static_cast<std::size_t>(o.end - o.p) + 1;
    const long long sv = static_cast<long long>(v);
BOLT_LOG_NONLITERAL_BEGIN
    int w = 0;
    if (n_stars == 0) w = std::snprintf(o.p, room, spec, sv);
    else if (n_stars == 1) w = std::snprintf(o.p, room, spec, stars[0], sv);
    else w = std::snprintf(o.p, room, spec, stars[0], stars[1], sv);
BOLT_LOG_NONLITERAL_END
    o.advance(w);
}

void emit_float(Out& o, Spec& s, const int* stars, std::uint32_t n_stars, const Arg& a) noexcept {
    double d;
    std::memcpy(&d, &a.bits, 8);
    char spec[32];
    std::memcpy(spec, s.head, s.head_n);
    spec[s.head_n] = s.conv;
    spec[s.head_n + 1] = '\0';
    const std::size_t room = static_cast<std::size_t>(o.end - o.p) + 1;
BOLT_LOG_NONLITERAL_BEGIN
    int w = 0;
    if (n_stars == 0) w = std::snprintf(o.p, room, spec, d);
    else if (n_stars == 1) w = std::snprintf(o.p, room, spec, stars[0], d);
    else w = std::snprintf(o.p, room, spec, stars[0], stars[1], d);
BOLT_LOG_NONLITERAL_END
    o.advance(w);
}

void emit_str(Out& o, const Spec& s, const int* stars, std::uint32_t n_stars, const Arg& a) noexcept {
    std::uint32_t len = a.len;
    if (s.star_prec) {
        const int prec = stars[n_stars - 1];
        if (prec >= 0 && static_cast<std::uint32_t>(prec) < len) len = static_cast<std::uint32_t>(prec);
    }
    o.raw(a.str != nullptr ? a.str : "", len);
}

}  // namespace

void format_message(const Record& r, char* out, std::size_t cap, std::size_t* out_n) noexcept {
    assert(out != nullptr && cap > 1);
    assert(r.fmt != nullptr);
    Out o{out, out + cap - 1};
    Reader rd{r.payload, r.payload + r.payload_len};
    for (const char* f = r.fmt; *f != '\0';) {
        if (*f != '%') { o.ch(*f++); continue; }
        ++f;
        if (*f == '%') { o.ch('%'); ++f; continue; }
        Spec s;
        const char* after = parse_spec(f, &s);
        if (after == nullptr) break;
        f = after;
        int stars[2] = {0, 0};
        std::uint32_t n_stars = 0;
        bool ok = true;
        for (std::uint32_t i = 0; i < (s.star_width ? 1u : 0u) + (s.star_prec ? 1u : 0u); ++i) {
            Arg a;
            ok = next_arg(rd, &a) && ok;
            stars[n_stars++] = static_cast<int>(a.bits);
        }
        Arg a;
        if (!ok || !next_arg(rd, &a)) { o.str("<truncated>"); break; }
        if (s.conv == 's' && a.tag == kArgStr) emit_str(o, s, stars, n_stars, a);
        else if (s.conv == 'p') { char b[24]; o.advance(0); std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(a.bits)); o.str(b); }
        else if (s.conv == 'c') o.ch(static_cast<char>(a.bits));
        else if (std::strchr("diuxXo", s.conv) != nullptr) emit_int(o, s, stars, n_stars, a);
        else if (std::strchr("eEfFgGaA", s.conv) != nullptr) emit_float(o, s, stars, n_stars, a);
        else o.str("<?>");
    }
    while (o.p > out && (o.p[-1] == '\n' || o.p[-1] == '\r')) --o.p;
    *o.p = '\0';
    *out_n = static_cast<std::size_t>(o.p - out);
}

namespace {

void put_time(Out& o, std::uint64_t ts_ns) noexcept {
    const std::time_t secs = static_cast<std::time_t>(ts_ns / 1000000000ull);
    const unsigned ms = static_cast<unsigned>((ts_ns / 1000000ull) % 1000u);
    struct tm tm_utc;
#if defined(_WIN32)
    gmtime_s(&tm_utc, &secs);
#else
    gmtime_r(&secs, &tm_utc);
#endif
    char b[40];
    const int n = std::snprintf(b, sizeof(b), "%04d-%02d-%02dT%02d:%02d:%02d.%03uZ", tm_utc.tm_year + 1900,
                                tm_utc.tm_mon + 1, tm_utc.tm_mday, tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec, ms);
    o.raw(b, static_cast<std::size_t>(n));
}

void json_str(Out& o, const char* s, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '"' || c == '\\') { o.ch('\\'); o.ch(static_cast<char>(c)); }
        else if (c == '\n') o.str("\\n");
        else if (c == '\r') o.str("\\r");
        else if (c == '\t') o.str("\\t");
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o.str(b); }
        else o.ch(static_cast<char>(c));
    }
}

void put_trace(Out& o, const Record& r, const char* trace_key, const char* span_key) noexcept {
    char b[64];
    if (r.trace_hi != 0) std::snprintf(b, sizeof(b), "%016llx%016llx", static_cast<unsigned long long>(r.trace_hi), static_cast<unsigned long long>(r.trace_lo));
    else std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(r.trace_lo));
    o.str(trace_key); o.str(b);
    if (r.span != 0) {
        std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(r.span));
        o.str(span_key); o.str(b);
    }
}

const char* base_name(const char* path) noexcept {
    const char* slash = std::strrchr(path, '/');
    return slash != nullptr ? slash + 1 : path;
}

void line_text(Out& o, const Record& r, const char* module, const char* msg, std::size_t msg_n) noexcept {
    char b[24];
    put_time(o, r.ts_ns);
    std::snprintf(b, sizeof(b), " %-5s [", level_name(static_cast<Level>(r.level)));
    o.str(b); o.str(module); o.str("] ");
    if ((r.flags & kFlagEvent) != 0) { o.str(r.fmt); o.ch(' '); }
    o.raw(msg, msg_n);
    if (r.trace_lo != 0 || r.trace_hi != 0) put_trace(o, r, " trace=", " span=");
    if (r.level >= static_cast<std::uint8_t>(Level::Warn) && r.file != nullptr) {
        o.str(" ("); o.str(base_name(r.file));
        std::snprintf(b, sizeof(b), ":%u)", r.line);
        o.str(b);
    }
    o.ch('\n');
}

void line_json(Out& o, const Record& r, const char* module, const char* service, const char* msg, std::size_t msg_n) noexcept {
    char b[48];
    o.str("{\"ts\":\""); put_time(o, r.ts_ns);
    o.str("\",\"level\":\""); o.str(level_name(static_cast<Level>(r.level)));
    o.str("\",\"service\":\""); o.str(service);
    o.str("\",\"module\":\""); o.str(module); o.ch('"');
    if ((r.flags & kFlagEvent) != 0) {
        o.str(",\"event\":\""); o.str(r.fmt); o.ch('"');
        if (msg_n != 0) { o.ch(','); o.raw(msg, msg_n); }
    } else {
        o.str(",\"msg\":\""); json_str(o, msg, msg_n); o.ch('"');
    }
    std::snprintf(b, sizeof(b), ",\"tid\":%u", r.tid);
    o.str(b);
    if (r.trace_lo != 0 || r.trace_hi != 0) {
        char t[64];
        Out to{t, t + sizeof(t) - 1};
        put_trace(to, r, "", "\",\"span_id\":\"");
        *to.p = '\0';
        o.str(",\"trace_id\":\""); o.str(t); o.ch('"');
    }
    if ((r.flags & kFlagTruncated) != 0) o.str(",\"truncated\":true");
    if (r.file != nullptr) {
        o.str(",\"file\":\""); o.str(base_name(r.file));
        std::snprintf(b, sizeof(b), "\",\"line\":%u", r.line);
        o.str(b);
    }
    o.str("}\n");
}

}  // namespace

std::size_t format_line(const Record& r, Format format, const char* module, const char* service,
                        char* out, std::size_t cap) noexcept {
    assert(cap >= kRecordBytes);
    assert(out != nullptr);
    char msg[kFormatBytes];
    std::size_t msg_n = 0;
    if ((r.flags & kFlagEvent) != 0) {
        msg_n = r.payload_len < sizeof(msg) ? r.payload_len : sizeof(msg) - 1;
        std::memcpy(msg, r.payload, msg_n);
    } else {
        format_message(r, msg, sizeof(msg), &msg_n);
    }
    Out o{out, out + cap - 1};
    if (format == Format::Json) line_json(o, r, module, service, msg, msg_n);
    else line_text(o, r, module, msg, msg_n);
    if (o.p > out && o.p[-1] != '\n') o.p[-1] = '\n';
    return static_cast<std::size_t>(o.p - out);
}

}  // namespace bolt::log::detail

#endif  // BOLT_OBS_LOG
