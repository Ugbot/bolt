// bolt_csv_spark.h -- Spark 4 CSV schema inference for one literal line (schema_of_csv).
//
// Splits a record the way univocity does under Spark's defaults (sep ',', quote '"') and classifies each
// field with CSVInferSchema's rules, but ONLY for the classes whose answer is a closed form: empty, integer,
// decimal-with-no-fraction past long, fractional double, boolean, NaN / Infinity, and text with no digit.
// A field whose type turns on Spark's lenient date / timestamp / hex / suffix parsers is reported as
// kCsvUnsupported, never guessed.
//
// RULES: no exceptions, no heap, noexcept, bounded loops; output goes to a caller buffer.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {
namespace kernels {
namespace csvspark {

enum class CsvType : std::uint8_t { Str, Int, Long, Decimal, Double, Bool };

constexpr int kCsvOk = 1;
constexpr int kCsvUnsupported = 0;   // a form this kernel will not guess
constexpr int kCsvOverflow = -1;     // field count or output past the caller's budget

struct CsvField {
    CsvType       type;
    std::uint8_t  precision;   // Decimal only
};

namespace detail {

inline bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

inline bool ieq(const char* s, std::uint32_t n, const char* lit) noexcept {
    const std::uint32_t m = static_cast<std::uint32_t>(std::strlen(lit));
    if (n != m) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        char c = s[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c != lit[i]) return false;
    }
    return true;
}

// Integer text [+-]?digits: Int / Long / Decimal(p,0) / Double (past 38 digits), as Spark's chain does.
inline int classify_integer(const char* s, std::uint32_t n, CsvField* out) noexcept {
    std::uint32_t i = (s[0] == '+' || s[0] == '-') ? 1u : 0u;
    const bool neg = s[0] == '-';
    while (i + 1 < n && s[i] == '0') ++i;                  // leading zeros do not count toward precision
    const std::uint32_t digits = n - i;
    if (digits <= 9) { out->type = CsvType::Int; return kCsvOk; }
    if (digits <= 19) {
        unsigned __int128 v = 0;
        for (std::uint32_t k = i; k < n; ++k) v = v * 10u + static_cast<unsigned>(s[k] - '0');
        const unsigned __int128 imax = neg ? 2147483648u : 2147483647u;
        const unsigned __int128 lmax = neg ? (static_cast<unsigned __int128>(1) << 63) : (static_cast<unsigned __int128>(1) << 63) - 1u;
        if (v <= imax) { out->type = CsvType::Int; return kCsvOk; }
        if (v <= lmax) { out->type = CsvType::Long; return kCsvOk; }
    }
    if (digits <= 38) { out->type = CsvType::Decimal; out->precision = static_cast<std::uint8_t>(digits); return kCsvOk; }
    out->type = CsvType::Double;
    return kCsvOk;
}

// [+-]?digits*.digits+ : a fraction makes the scale positive, which Spark answers as DOUBLE.
inline bool is_fractional(const char* s, std::uint32_t n) noexcept {
    std::uint32_t i = (s[0] == '+' || s[0] == '-') ? 1u : 0u;
    while (i < n && is_digit(s[i])) ++i;
    if (i >= n || s[i] != '.') return false;
    const std::uint32_t frac = ++i;
    while (i < n && is_digit(s[i])) ++i;
    return i == n && i > frac;
}

inline bool has_digit(const char* s, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) if (is_digit(s[i])) return true;
    return false;
}

}  // namespace detail

// One unquoted-or-unescaped field's type. kCsvUnsupported for a form that needs Spark's lenient parsers.
inline int csv_classify(const char* s, std::uint32_t n, CsvField* out) noexcept {
    assert(s != nullptr && out != nullptr);
    out->type = CsvType::Str;
    out->precision = 0;
    if (n == 0) return kCsvOk;                                       // empty = null value = NullType -> STRING
    if (detail::ieq(s, n, "true") || detail::ieq(s, n, "false")) { out->type = CsvType::Bool; return kCsvOk; }
    if (n == 3 && std::memcmp(s, "NaN", 3) == 0) { out->type = CsvType::Double; return kCsvOk; }
    if ((n == 8 && std::memcmp(s, "Infinity", 8) == 0) || (n == 9 && std::memcmp(s, "-Infinity", 9) == 0)) {
        out->type = CsvType::Double;
        return kCsvOk;
    }
    if (detail::ieq(s, n, "nan") || detail::ieq(s, n, "infinity") || detail::ieq(s, n, "+infinity") ||
        detail::ieq(s, n, "-infinity")) return kCsvUnsupported;      // another spelling: Spark's answer depends on the parser
    if (!detail::has_digit(s, n)) {
        for (std::uint32_t i = 0; i < n; ++i) if (s[i] == '.' ) return kCsvUnsupported;
        return kCsvOk;                                               // text with no digit and no '.' is STRING
    }
    std::uint32_t i = (s[0] == '+' || s[0] == '-') ? 1u : 0u;
    std::uint32_t j = i;
    while (j < n && detail::is_digit(s[j])) ++j;
    if (j == n && j > i) return detail::classify_integer(s, n, out);
    if (detail::is_fractional(s, n)) { out->type = CsvType::Double; return kCsvOk; }
    return kCsvUnsupported;                                          // dates, timestamps, 1., 1e3, 1d, 0x10, padded numbers ...
}

// Splits one record at ',' (quote '"', no embedded record separator) into fields and classifies each.
// kCsvUnsupported for CR / LF, a quote inside an unquoted field, text after a closing quote, or a backslash
// inside quotes; kCsvOverflow past `max_fields`. `n_out` receives the field count.
inline int csv_infer_line(const char* s, std::uint32_t n, CsvField* fields, std::uint32_t max_fields,
                          std::uint32_t* n_out, char* scratch, std::uint32_t scratch_cap) noexcept {
    assert(fields != nullptr && n_out != nullptr && scratch != nullptr);
    std::uint32_t count = 0, i = 0;
    for (std::uint32_t k = 0; k < n; ++k) if (s[k] == '\n' || s[k] == '\r') return kCsvUnsupported;
    if (n > scratch_cap) return kCsvOverflow;
    for (;;) {
        if (count >= max_fields) return kCsvOverflow;
        std::uint32_t len = 0;
        if (i < n && s[i] == '"') {
            ++i;
            for (;;) {
                if (i >= n) return kCsvUnsupported;                  // unterminated quote
                if (s[i] == '\\') return kCsvUnsupported;
                if (s[i] == '"') {
                    if (i + 1 < n && s[i + 1] == '"') { scratch[len++] = '"'; i += 2; continue; }
                    ++i;
                    break;
                }
                scratch[len++] = s[i++];
            }
            if (i < n && s[i] != ',') return kCsvUnsupported;
        } else {
            while (i < n && s[i] != ',') {
                if (s[i] == '"') return kCsvUnsupported;
                scratch[len++] = s[i++];
            }
        }
        const int rc = csv_classify(scratch, len, &fields[count]);
        if (rc != kCsvOk) return rc;
        ++count;
        if (i >= n) break;
        ++i;                                                         // the separator; a trailing one opens an empty field
        if (i == n) { if (count >= max_fields) return kCsvOverflow; fields[count].type = CsvType::Str; fields[count].precision = 0; ++count; break; }
    }
    *n_out = count;
    return kCsvOk;
}

// "STRUCT<_c0: INT, _c1: STRING>" into out[0..cap); returns the length, or kCsvOverflow.
inline int csv_schema_text(const CsvField* fields, std::uint32_t count, char* out, std::uint32_t cap) noexcept {
    assert(fields != nullptr && out != nullptr && count > 0);
    std::uint32_t n = 0;
    auto put = [&](const char* t) noexcept -> bool {
        const std::uint32_t m = static_cast<std::uint32_t>(std::strlen(t));
        if (n + m >= cap) return false;
        std::memcpy(out + n, t, m);
        n += m;
        return true;
    };
    if (!put("STRUCT<")) return kCsvOverflow;
    for (std::uint32_t k = 0; k < count; ++k) {
        char nm[48];
        std::uint32_t d = 0;
        char dig[12];
        for (std::uint32_t v = k; ; v /= 10u) { dig[d++] = static_cast<char>('0' + v % 10u); if (v < 10u) break; }
        std::uint32_t m = 0;
        if (k != 0) { nm[m++] = ','; nm[m++] = ' '; }
        nm[m++] = '_'; nm[m++] = 'c';
        while (d > 0) nm[m++] = dig[--d];
        nm[m++] = ':'; nm[m++] = ' ';
        nm[m] = '\0';
        if (!put(nm)) return kCsvOverflow;
        const CsvField& f = fields[k];
        bool ok = true;
        switch (f.type) {
            case CsvType::Int: ok = put("INT"); break;
            case CsvType::Long: ok = put("BIGINT"); break;
            case CsvType::Double: ok = put("DOUBLE"); break;
            case CsvType::Bool: ok = put("BOOLEAN"); break;
            case CsvType::Str: ok = put("STRING"); break;
            case CsvType::Decimal: {
                char t[16];
                t[0] = 'D'; t[1] = 'E'; t[2] = 'C'; t[3] = 'I'; t[4] = 'M'; t[5] = 'A'; t[6] = 'L'; t[7] = '(';
                std::uint32_t q = 8;
                if (f.precision >= 10) t[q++] = static_cast<char>('0' + f.precision / 10);
                t[q++] = static_cast<char>('0' + f.precision % 10);
                t[q++] = ','; t[q++] = '0'; t[q++] = ')'; t[q] = '\0';
                ok = put(t);
                break;
            }
        }
        if (!ok) return kCsvOverflow;
    }
    if (!put(">")) return kCsvOverflow;
    out[n] = '\0';
    return static_cast<int>(n);
}

}  // namespace csvspark
}  // namespace kernels
}  // namespace bolt
