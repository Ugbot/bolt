// OTLP metric and label names in Prometheus's spelling, and the native histogram bytes.
// See bolt_otlp_metrics.h.

#include "bolt/ingest/bolt_otlp_metrics.h"

#include <cassert>
#include <cstring>

namespace bolt::ingest::otlp {

namespace {

struct UnitName { std::string_view unit, name; };
constexpr UnitName kUnits[] = {
    {"d", "days"}, {"h", "hours"}, {"min", "minutes"}, {"s", "seconds"}, {"ms", "milliseconds"}, {"us", "microseconds"},
    {"ns", "nanoseconds"}, {"By", "bytes"}, {"KiBy", "kibibytes"}, {"MiBy", "mebibytes"}, {"GiBy", "gibibytes"},
    {"TiBy", "tibibytes"}, {"KBy", "kilobytes"}, {"MBy", "megabytes"}, {"GBy", "gigabytes"}, {"TBy", "terabytes"},
    {"m", "meters"}, {"V", "volts"}, {"A", "amperes"}, {"J", "joules"}, {"W", "watts"}, {"g", "grams"},
    {"Cel", "celsius"}, {"Hz", "hertz"}, {"%", "percent"}, {"1", ""},
};
constexpr UnitName kPerUnits[] = {
    {"s", "second"}, {"m", "minute"}, {"h", "hour"}, {"d", "day"}, {"w", "week"}, {"mo", "month"}, {"y", "year"},
};

bool name_char(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ':';
}

// A unit with its {annotation} taken off and blanks trimmed.
std::string_view clean_unit(std::string_view u) noexcept {
    const std::size_t brace = u.find('{');
    if (brace != std::string_view::npos) u = u.substr(0, brace);
    while (!u.empty() && u.back() == ' ') u.remove_suffix(1);
    while (!u.empty() && u.front() == ' ') u.remove_prefix(1);
    return u;
}

template <std::size_t N>
std::string_view unit_word(std::string_view u, const UnitName (&table)[N]) noexcept {
    for (const UnitName& x : table) if (x.unit == u) return x.name;
    return u;
}

struct Tokens {
    std::string_view t[kNameTokens];
    std::uint32_t    n = 0;
    bool             ok = true;
};

// The words of `s`; with `unique`, a word already present is not added again.
void add_tokens(Tokens* tk, std::string_view s, bool unique) noexcept {
    std::size_t i = 0;
    while (i < s.size()) {   // bounded: s.size()
        while (i < s.size() && !name_char(s[i])) ++i;
        const std::size_t b = i;
        while (i < s.size() && name_char(s[i])) ++i;
        if (i == b) continue;
        const std::string_view word = s.substr(b, i - b);
        bool seen = false;
        for (std::uint32_t k = 0; unique && k < tk->n && !seen; ++k) seen = tk->t[k] == word;
        if (seen) continue;
        if (tk->n == kNameTokens) { tk->ok = false; return; }
        tk->t[tk->n++] = word;
    }
}

}  // namespace

bool metric_name(std::string_view name, std::string_view unit, int kind, char* out, std::size_t cap,
                 std::size_t* len) noexcept {
    assert(out != nullptr && len != nullptr && kind >= 0 && kind <= 2);
    *len = 0;
    Tokens tk;
    add_tokens(&tk, name, false);
    const std::size_t slash = unit.find('/');
    const std::string_view main = clean_unit(unit.substr(0, slash));
    const std::string_view per = slash == std::string_view::npos ? std::string_view() : clean_unit(unit.substr(slash + 1));
    if (!main.empty()) add_tokens(&tk, unit_word(main, kUnits), true);
    if (!per.empty()) { add_tokens(&tk, "per", true); add_tokens(&tk, unit_word(per, kPerUnits), true); }
    if (kind == 0 && main == "1" && per.empty()) add_tokens(&tk, "ratio", true);
    if (kind == 1) {
        std::uint32_t w = 0;
        for (std::uint32_t i = 0; i < tk.n; ++i) if (tk.t[i] != "total") tk.t[w++] = tk.t[i];   // bounded: kNameTokens
        tk.n = w;
        if (tk.n == kNameTokens) tk.ok = false; else tk.t[tk.n++] = "total";
    }
    if (!tk.ok) return false;
    std::size_t n = 0;
    if (tk.n != 0 && tk.t[0][0] >= '0' && tk.t[0][0] <= '9') { if (cap == 0) return false; out[n++] = '_'; }
    for (std::uint32_t i = 0; i < tk.n; ++i) {   // bounded: kNameTokens
        if (cap - n < tk.t[i].size() + 1u) return false;
        if (i != 0) out[n++] = '_';
        std::memcpy(out + n, tk.t[i].data(), tk.t[i].size());
        n += tk.t[i].size();
    }
    *len = n;
    return true;
}

std::size_t label_name(std::string_view name, char* out, std::size_t cap) noexcept {
    assert(out != nullptr && cap >= name.size() + 4u);
    (void)cap;
    const auto tr = [](char c) noexcept { return (name_char(c) && c != ':') || c == '_' ? c : '_'; };
    std::size_t n = 0;
    const char c0 = name.empty() ? '\0' : tr(name[0]);
    if (c0 >= '0' && c0 <= '9') { std::memcpy(out, "key_", 4); n = 4; }
    else if (c0 == '_' && (name.size() == 1 || tr(name[1]) != '_')) { std::memcpy(out, "key", 3); n = 3; }
    for (const char c : name) out[n++] = tr(c);   // bounded: name.size()
    return n;
}

bool hist_decode(std::string_view raw, HistHead* head, double* pos, double* neg, std::uint32_t cap) noexcept {
    assert(head != nullptr && pos != nullptr && neg != nullptr);
    if (raw.size() < sizeof(HistHead)) return false;
    std::memcpy(head, raw.data(), sizeof(HistHead));
    if (head->n_pos > cap || head->n_neg > cap) return false;
    const std::size_t want = sizeof(HistHead) + (static_cast<std::size_t>(head->n_pos) + head->n_neg) * 8u;
    if (raw.size() != want) return false;
    if (head->n_pos != 0) std::memcpy(pos, raw.data() + sizeof(HistHead), head->n_pos * 8u);
    if (head->n_neg != 0) std::memcpy(neg, raw.data() + sizeof(HistHead) + head->n_pos * 8u, head->n_neg * 8u);
    return true;
}

bool hex_decode(std::string_view hex, char* out) noexcept {
    assert(out != nullptr || hex.empty());
    if (hex.size() % 2u != 0) return false;
    const auto nib = [](char c) noexcept -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < hex.size(); i += 2) {   // bounded: hex.size()
        const int hi = nib(hex[i]), lo = nib(hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i / 2] = static_cast<char>((hi << 4) | lo);
    }
    return true;
}

}  // namespace bolt::ingest::otlp
