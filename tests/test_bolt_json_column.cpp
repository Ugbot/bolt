// test_bolt_json_column.cpp — columnar JSON path extraction, differential
// against the per-document json_path_lookup over random JSON.

#include "bolt/kernels/bolt_json_column.h"

#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

using namespace bolt;
using namespace bolt::kernels::json;

namespace {

struct Col {
    std::vector<std::string> rows;
    std::vector<StringView> sv;
    std::vector<char> spill;
    void build() {
        sv.assign(rows.size(), StringView{});
        size_t total = 0;
        for (const auto& r : rows) if (r.size() > 12) total += r.size();
        spill.assign(total + 1, 0);
        size_t off = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            std::memset(&sv[i], 0, sizeof(StringView));
            sv[i].length = static_cast<uint32_t>(r.size());
            if (r.size() <= 12) {
                std::memcpy(sv[i].prefix, r.data(), r.size() > 4 ? 4 : r.size());
                if (r.size() > 4) std::memcpy(sv[i].inline_data, r.data() + 4, r.size() - 4);
            } else {
                std::memcpy(spill.data() + off, r.data(), r.size());
                std::memcpy(sv[i].prefix, r.data(), 4);
                sv[i].ref.buf_idx = 0;
                sv[i].ref.offset = static_cast<uint32_t>(off);
                off += r.size();
            }
        }
    }
};

std::string rnd_scalar(std::mt19937& g) {
    switch (g() % 9) {
        case 0: return std::to_string(static_cast<int64_t>(g()) - 2000000000LL);
        case 1: return "404";
        case 2: return "\"404\"";
        case 3: return "4.5";
        case 4: return "1e3";
        case 5: return "null";
        case 6: return "true";
        case 7: return "9223372036854775807";
        default: return "\"a \\\" b \\u00e9 }\"";
    }
}

std::string rnd_doc(std::mt19937& g, int depth) {
    static const char* keys[] = {"status", "a", "b", "attrs", "x\\\"y", "k1", "size"};
    const unsigned t = g() % 12;
    if (t == 0) return "";                      // empty
    if (t == 1) return "{\"status\":40";        // truncated
    if (t == 2) return "[1,2,3]";
    if (t == 3) return "not json";
    std::string s = "{";
    const int n = static_cast<int>(g() % 6);
    for (int i = 0; i < n; ++i) {
        if (i) s += g() % 4 == 0 ? " , " : ",";
        s += std::string("\"") + keys[g() % 7] + "\":";
        if (depth < 2 && g() % 4 == 0) s += rnd_doc(g, depth + 1);
        else if (g() % 6 == 0) s += "[" + rnd_scalar(g) + "," + rnd_scalar(g) + "]";
        else if (g() % 9 == 0) s += "\"" + std::string(100 + g() % 200, 'z') + "\"";   // huge value
        else s += rnd_scalar(g);
    }
    return s + "}";
}

bool ref_i64(const JsonPath& p, const std::string& d, int64_t* v) {
    JsonSpan s{};
    return json_path_lookup_generic(&p, d.data(), static_cast<uint32_t>(d.size()), &s) && json_span_to_i64(s, v);
}

}  // namespace

TEST(JsonColumn, DifferentialAgainstPerDocumentLookup) {
    std::mt19937 g(20261116);
    const char* paths[] = {"$.status", "status", "$.a.status", "$.attrs.a", "$.a[1]", "$"};
    for (const char* ps : paths) {
        JsonPath p{};
        ASSERT_TRUE(json_path_compile(ps, static_cast<uint32_t>(std::strlen(ps)), &p));
        Col c;
        for (int i = 0; i < 6000; ++i) {
            std::string d = rnd_doc(g, 0);
            if (i % 3 == 0 && !d.empty()) {                 // damage it: drop, insert, or swap a byte
                const size_t at = g() % d.size();
                static const char junk[] = "{}[]\",: \\x1";
                switch (g() % 3) {
                    case 0: d.erase(at, 1); break;
                    case 1: d.insert(d.begin() + static_cast<long>(at), junk[g() % (sizeof(junk) - 1)]); break;
                    default: d[at] = junk[g() % (sizeof(junk) - 1)]; break;
                }
            }
            if (i % 11 == 0) d = std::string(g() % 3, ' ') + d + std::string(g() % 3, '\n');
            c.rows.push_back(d);
        }
        c.build();
        const int64_t n = static_cast<int64_t>(c.rows.size());
        std::vector<int64_t> out(n); std::vector<double> fo(n); std::vector<int64_t> ex(n);
        std::vector<uint8_t> valid((n + 7) / 8, 0xAA), fvalid((n + 7) / 8, 0x55);
        const int64_t nulls = json_column_extract_i64(&p, c.sv.data(), c.spill.data(), nullptr, n, out.data(), valid.data());
        const int64_t fnulls = json_column_extract_f64(&p, c.sv.data(), c.spill.data(), nullptr, n, fo.data(), fvalid.data());
        json_column_exists(&p, c.sv.data(), c.spill.data(), nullptr, n, ex.data());
        int64_t ref_nulls = 0, ref_fnulls = 0;
        for (int64_t i = 0; i < n; ++i) {
            int64_t rv = 0; const bool rok = ref_i64(p, c.rows[i], &rv);
            ASSERT_EQ(((valid[i >> 3] >> (i & 7)) & 1) != 0, rok) << ps << " row " << i << " " << c.rows[i];
            ASSERT_EQ(out[i], rok ? rv : 0);
            ref_nulls += rok ? 0 : 1;
            JsonSpan s{}; double fv = 0;
            const bool found = json_path_lookup_generic(&p, c.rows[i].data(), static_cast<uint32_t>(c.rows[i].size()), &s);
            const bool fok = found && json_span_to_f64(s, &fv);
            ASSERT_EQ(((fvalid[i >> 3] >> (i & 7)) & 1) != 0, fok) << ps << " row " << i;
            if (fok) ASSERT_EQ(fo[i], fv);
            ref_fnulls += fok ? 0 : 1;
            ASSERT_EQ(ex[i], found ? 1 : 0);
        }
        EXPECT_EQ(nulls, ref_nulls);
        EXPECT_EQ(fnulls, ref_fnulls);
    }
}

TEST(JsonColumn, SelectionVectorAndTailBits) {
    JsonPath p{};
    ASSERT_TRUE(json_path_compile("$.status", 8, &p));
    Col c;
    for (int i = 0; i < 37; ++i) c.rows.push_back("{\"status\":" + std::to_string(i) + ",\"pad\":\"xxxxxxxxxxxxxxxxxxxx\"}");
    c.build();
    std::vector<int32_t> sel = {36, 3, 3, 0, 17};
    std::vector<int64_t> out(sel.size()); std::vector<uint8_t> valid(1, 0);
    EXPECT_EQ(0, json_column_extract_i64(&p, c.sv.data(), c.spill.data(), sel.data(), 5, out.data(), valid.data()));
    EXPECT_EQ(valid[0], 0x1F);
    EXPECT_EQ(out, (std::vector<int64_t>{36, 3, 3, 0, 17}));
    EXPECT_EQ(0, json_column_extract_i64(&p, c.sv.data(), c.spill.data(), nullptr, 0, out.data(), valid.data()));
}

TEST(JsonColumn, DepthLimitMatchesGenericAtTheEdge) {
    JsonPath p{};
    ASSERT_TRUE(json_path_compile("$.k", 3, &p));
    for (int depth = 28; depth <= 36; ++depth) {
        for (int where = 0; where < 2; ++where) {            // deep value before / after the sought key
            std::string deep(static_cast<size_t>(depth), '[');
            deep += std::string(static_cast<size_t>(depth), ']');
            const std::string d = where == 0 ? "{\"a\":" + deep + ",\"k\":7}" : "{\"k\":7,\"a\":" + deep + "}";
            JsonSpan a{}, b{};
            const bool ra = json_path_lookup(&p, d.data(), static_cast<uint32_t>(d.size()), &a);
            const bool rb = json_path_lookup_generic(&p, d.data(), static_cast<uint32_t>(d.size()), &b);
            ASSERT_EQ(ra, rb) << "depth " << depth << " where " << where;
            if (ra) { ASSERT_EQ(a.len, b.len); ASSERT_EQ(a.p, b.p); }
        }
    }
}
