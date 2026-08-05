// bolt/lakehouse/delta_generated_column.cpp — skeleton expression evaluator.
// Supported: identity (`col`), YEAR(col), MONTH(col), `col + N`.

#include "bolt/lakehouse/delta/generated_column.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace bj = bolt::parse::json;

namespace {

inline bool tok_eq(const bj::StructuralIndex* idx, int32_t cur,
                   const char* lit) noexcept {
    assert(idx != nullptr && lit != nullptr);
    if (cur < 0 || cur >= idx->token_count) return false;
    const bj::Token& t = idx->tokens[cur];
    if (t.type != bj::TokenType::Key && t.type != bj::TokenType::String)
        return false;
    const size_t n = std::strlen(lit);
    if (t.length != static_cast<int32_t>(n)) return false;
    return std::memcmp(idx->src + t.start, lit, n) == 0;
}

void copy_tok(const bj::StructuralIndex* idx, int32_t cur, char* dst,
              uint32_t cap) noexcept {
    assert(idx != nullptr && dst != nullptr);
    assert(cap > 0u);
    const bj::Token& t = idx->tokens[cur];
    const uint32_t n = t.length > static_cast<int32_t>(cap - 1u)
                           ? cap - 1u
                           : static_cast<uint32_t>(t.length);
    if (n > 0) std::memcpy(dst, idx->src + t.start, n);
    dst[n] = '\0';
}

// `bj::iter_skip_to_close` counts depth FROM the token under the cursor and
// already advances once for a scalar, so it must be called ON the Begin token.
// Advancing past Begin first leaves the cursor INSIDE the value and silently
// truncates the enclosing parse — see the long note in iceberg_metadata.cpp
// (G2FEAT-125), which is where that bug was finally caught.
bool skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    return bj::iter_skip_to_close(it);
}

void rtrim(char* s) noexcept {
    assert(s != nullptr);
    size_t n = std::strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t')) {
        s[n - 1] = '\0'; --n;
    }
}

void parse_expr(const char* expr_in, char* col, uint32_t col_cap,
                GenExprKind* kind, int64_t* addend) noexcept {
    assert(expr_in != nullptr && col != nullptr && kind != nullptr);
    assert(addend != nullptr);
    *kind = GenExprKind::kUnknown;
    *addend = 0;
    col[0] = '\0';
    char buf[kLakeMaxExprBytes];
    size_t n = std::strlen(expr_in);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1u;
    std::memcpy(buf, expr_in, n);
    buf[n] = '\0';
    char* p = buf;
    while (*p == ' ' || *p == '\t') ++p;
    rtrim(p);
    auto try_fn = [&](const char* fn, GenExprKind k) noexcept -> bool {
        const size_t fl = std::strlen(fn);
        if (std::strncmp(p, fn, fl) != 0) return false;
        const char* q = p + fl;
        while (*q == ' ' || *q == '\t') ++q;
        if (*q != '(') return false;
        ++q;
        while (*q == ' ' || *q == '\t') ++q;
        const char* name_start = q;
        while (*q && *q != ')' && *q != ' ' && *q != '\t') ++q;
        const size_t nl = static_cast<size_t>(q - name_start);
        if (nl == 0 || nl >= col_cap) return false;
        std::memcpy(col, name_start, nl);
        col[nl] = '\0';
        *kind = k;
        return true;
    };
    if (try_fn("YEAR",  GenExprKind::kYearOfDate)) return;
    if (try_fn("MONTH", GenExprKind::kMonthOfDate)) return;
    const char* plus = std::strchr(p, '+');
    if (plus != nullptr) {
        const char* end = plus;
        while (end > p && (end[-1] == ' ' || end[-1] == '\t')) --end;
        const size_t nl = static_cast<size_t>(end - p);
        if (nl > 0 && nl < col_cap) {
            std::memcpy(col, p, nl);
            col[nl] = '\0';
            const char* num = plus + 1;
            while (*num == ' ' || *num == '\t') ++num;
            char* parse_end = nullptr;
            const long long v = std::strtoll(num, &parse_end, 10);
            if (parse_end != num) {
                *addend = static_cast<int64_t>(v);
                *kind = GenExprKind::kAddConst;
                return;
            }
        }
    }
    const char* q = p;
    while (*q && (std::isalnum(static_cast<unsigned char>(*q)) || *q == '_'))
        ++q;
    if (*q == '\0' && q != p) {
        const size_t nl = static_cast<size_t>(q - p);
        if (nl < col_cap) {
            std::memcpy(col, p, nl);
            col[nl] = '\0';
            *kind = GenExprKind::kIdentity;
        }
    }
}

bool parse_field_metadata(const bj::StructuralIndex* idx, bj::Iterator* it,
                          char* expr_out, uint32_t ecap) noexcept {
    assert(idx != nullptr && it != nullptr && expr_out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    uint32_t g2 = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && g2++ < 64u) {
        const int32_t ck = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, ck, "delta.generationExpression") &&
            bj::iter_peek(it) == bj::TokenType::String) {
            copy_tok(idx, it->cursor, expr_out, ecap);
            bj::iter_advance(it);
        } else {
            skip_value(it);
        }
    }
    return bj::iter_advance(it);
}

bool parse_field(const bj::StructuralIndex* idx, bj::Iterator* it,
                 GeneratedColumnSet* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return skip_value(it);
    bj::iter_advance(it);
    char name[kLakeMaxColName] = {0};
    char expr[kLakeMaxExprBytes] = {0};
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key_cur, "name") &&
            bj::iter_peek(it) == bj::TokenType::String) {
            copy_tok(idx, it->cursor, name, sizeof(name));
            bj::iter_advance(it);
        } else if (tok_eq(idx, key_cur, "metadata")) {
            parse_field_metadata(idx, it, expr, sizeof(expr));
        } else {
            skip_value(it);
        }
    }
    bj::iter_advance(it);
    if (name[0] == '\0' || expr[0] == '\0') return true;
    if (out->n_cols >= kDeltaMaxGenCols) return false;
    GeneratedColumn* g = &out->cols[out->n_cols];
    std::memset(g, 0, sizeof(*g));
    std::memcpy(g->output_name, name, sizeof(g->output_name));
    parse_expr(expr, g->input_name, sizeof(g->input_name), &g->kind,
               &g->const_addend);
    if (g->kind != GenExprKind::kUnknown) ++out->n_cols;
    return true;
}

}  // namespace

bool delta_generated_cols_build(const DeltaMetadata* md, Arena* scratch,
                                GeneratedColumnSet* out) noexcept {
    assert(md != nullptr && scratch != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (md->schema_len == 0) return true;
    bj::StructuralIndex idx{};
    if (!bj::build_index(reinterpret_cast<const uint8_t*>(md->schema_string),
                          static_cast<int32_t>(md->schema_len),
                          scratch, &idx)) {
        return true;
    }
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return true;
    if (bj::iter_peek(&it) != bj::TokenType::BeginObject) return true;
    bj::iter_advance(&it);
    uint32_t guard = 0;
    while (bj::iter_peek(&it) == bj::TokenType::Key && guard++ < 64u) {
        const int32_t key_cur = it.cursor;
        bj::iter_advance(&it);
        if (tok_eq(&idx, key_cur, "fields") &&
            bj::iter_peek(&it) == bj::TokenType::BeginArray) {
            bj::iter_advance(&it);
            uint32_t g2 = 0;
            while (bj::iter_peek(&it) == bj::TokenType::BeginObject &&
                   g2++ < kDeltaMaxGenCols * 4u) {
                if (!parse_field(&idx, &it, out)) return false;
            }
            while (bj::iter_peek(&it) != bj::TokenType::EndArray &&
                   g2++ < kDeltaMaxGenCols * 8u) {
                if (!skip_value(&it)) break;
            }
            bj::iter_advance(&it);
        } else {
            skip_value(&it);
        }
    }
    return true;
}

namespace {

// Civil-from-days (Howard Hinnant). Date32 = days since 1970-01-01.
void civil_from_days(int64_t z, int32_t* y, int32_t* m, int32_t* d) noexcept {
    assert(y != nullptr && m != nullptr && d != nullptr);
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const uint32_t doe = static_cast<uint32_t>(z - era * 146097);
    const uint32_t yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    const int64_t  yy = static_cast<int64_t>(yoe) + era * 400;
    const uint32_t doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    const uint32_t mp = (5u * doy + 2u) / 153u;
    const uint32_t dd = doy - (153u * mp + 2u) / 5u + 1u;
    const uint32_t mm = mp < 10u ? mp + 3u : mp - 9u;
    *y = static_cast<int32_t>(yy + (mm <= 2u ? 1 : 0));
    *m = static_cast<int32_t>(mm);
    *d = static_cast<int32_t>(dd);
}

}  // namespace

bool delta_generated_eval_i64(const GeneratedColumn* g, int64_t input,
                              int64_t* out) noexcept {
    assert(g != nullptr && out != nullptr);
    switch (g->kind) {
        case GenExprKind::kIdentity: *out = input; return true;
        case GenExprKind::kYearOfDate: {
            int32_t y = 0, m = 0, d = 0;
            civil_from_days(input, &y, &m, &d);
            *out = y; return true;
        }
        case GenExprKind::kMonthOfDate: {
            int32_t y = 0, m = 0, d = 0;
            civil_from_days(input, &y, &m, &d);
            *out = m; return true;
        }
        case GenExprKind::kAddConst: *out = input + g->const_addend; return true;
        case GenExprKind::kUnknown: return false;
    }
    return false;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
