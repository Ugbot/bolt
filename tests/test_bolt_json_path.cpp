// test_bolt_json_path.cpp — value-asserted coverage for the bounded JSON
// path kernel (bolt_json_path.h): compile grammar, nested lookups, array
// indexing, missing keys, malformed documents (must return not-found —
// NEVER crash / never fabricate), escape decoding, and every documented
// cap (path depth, key length, skip nesting depth).

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/kernels/bolt_json_path.h"

#define CHK(c) do { if(!(c)){ std::fprintf(stderr,"FAIL %d %s\n",__LINE__,#c); std::exit(1);} } while(0)

using namespace bolt::kernels::json;

namespace {

bool lookup(const char* path, const char* doc, JsonSpan* out) {
    JsonPath p{};
    if (!json_path_compile(path, static_cast<std::uint32_t>(std::strlen(path)),
                           &p)) {
        return false;
    }
    return json_path_lookup(&p, doc,
                            static_cast<std::uint32_t>(std::strlen(doc)), out);
}

bool span_eq(const JsonSpan& sp, const char* want) {
    const std::size_t wl = std::strlen(want);
    return sp.len == wl && std::memcmp(sp.p, want, wl) == 0;
}

}  // namespace

int main() {
    // --- compile grammar -------------------------------------------------
    {
        JsonPath p{};
        CHK(json_path_compile("$", 1, &p) && p.seg_count == 0);
        CHK(json_path_compile("$.a.b", 5, &p) && p.seg_count == 2);
        CHK(p.segs[0].is_index == 0 && p.segs[0].key_len == 1 &&
            p.segs[0].key[0] == 'a');
        CHK(json_path_compile("$.a[3].b", 8, &p) && p.seg_count == 3);
        CHK(p.segs[1].is_index == 1 && p.segs[1].index == 3);
        // bare key: verbatim ONE key, even with a dot (PG -> semantics)
        CHK(json_path_compile("a.b", 3, &p) && p.seg_count == 1);
        CHK(p.segs[0].key_len == 3 && std::memcmp(p.segs[0].key, "a.b", 3) == 0);
        // bare index form ([2] = col -> 2)
        CHK(json_path_compile("[2]", 3, &p) && p.seg_count == 1 &&
            p.segs[0].is_index == 1 && p.segs[0].index == 2);
        // rejections: empty / `$.` / unclosed bracket / negative index /
        // wildcard / over-depth / over-long key
        CHK(!json_path_compile("", 0, &p));
        CHK(!json_path_compile("$.", 2, &p));
        CHK(!json_path_compile("$[1", 3, &p));
        CHK(!json_path_compile("$[-1]", 5, &p));
        CHK(!json_path_compile("$.*", 3, &p));
        CHK(!json_path_compile("$.a.b.c.d.e.f.g.h.i", 19, &p));  // 9 segs
        char longkey[80];
        std::memset(longkey, 'k', sizeof(longkey));
        longkey[0] = '$'; longkey[1] = '.';
        CHK(!json_path_compile(longkey, 60, &p));               // 58B key > 48
    }

    // --- nested lookups, value-asserted ---------------------------------
    const char* doc =
        "{\"a\": {\"b\": {\"c\": 42, \"s\": \"hi\"}, \"arr\": [10, 20, "
        "{\"x\": 1.5}]}, \"top\": true, \"nul\": null, "
        "\"neg\": -7, \"big\": 9223372036854775807}";
    {
        JsonSpan sp{};
        CHK(lookup("$.a.b.c", doc, &sp) && span_eq(sp, "42"));
        CHK(lookup("$.a.b.s", doc, &sp) && span_eq(sp, "\"hi\""));
        CHK(lookup("$.a.arr[1]", doc, &sp) && span_eq(sp, "20"));
        CHK(lookup("$.a.arr[2].x", doc, &sp) && span_eq(sp, "1.5"));
        CHK(lookup("$.top", doc, &sp) && span_eq(sp, "true"));
        CHK(lookup("$.nul", doc, &sp) && json_span_is_null(sp));
        CHK(lookup("$", doc, &sp) && sp.len == std::strlen(doc));
        // bare-key (-> operator) form
        CHK(lookup("top", doc, &sp) && span_eq(sp, "true"));
        // missing key / out-of-range index / type mismatch => not found
        CHK(!lookup("$.zzz", doc, &sp));
        CHK(!lookup("$.a.b.zzz", doc, &sp));
        CHK(!lookup("$.a.arr[3]", doc, &sp));
        CHK(!lookup("$.a.arr.c", doc, &sp));   // keying an array
        CHK(!lookup("$.top[0]", doc, &sp));    // indexing a scalar
    }

    // --- typed accessors -------------------------------------------------
    {
        JsonSpan sp{};
        std::int64_t iv = 0;
        double dv = 0.0;
        CHK(lookup("$.a.b.c", doc, &sp) && json_span_to_i64(sp, &iv) &&
            iv == 42);
        CHK(lookup("$.neg", doc, &sp) && json_span_to_i64(sp, &iv) &&
            iv == -7);
        CHK(lookup("$.big", doc, &sp) && json_span_to_i64(sp, &iv) &&
            iv == INT64_C(9223372036854775807));
        CHK(lookup("$.a.arr[2].x", doc, &sp));
        CHK(!json_span_to_i64(sp, &iv));       // 1.5 is not an integer
        CHK(json_span_to_f64(sp, &dv) && dv == 1.5);
        CHK(lookup("$.a.b.s", doc, &sp));
        CHK(!json_span_to_i64(sp, &iv));       // string is not a number
        CHK(!json_span_to_f64(sp, &dv));
        // int64 overflow rejected, not wrapped
        JsonSpan big{"92233720368547758080", 20};
        CHK(!json_span_to_i64(big, &iv));
        JsonSpan minv{"-9223372036854775808", 20};
        CHK(json_span_to_i64(minv, &iv) && iv == INT64_MIN);
    }

    // --- ->> unquote: escapes incl. \uXXXX + surrogate pair --------------
    {
        JsonSpan sp{};
        char buf[128];
        const char* d2 =
            "{\"s\": \"a\\\"b\\\\c\\n\\u0041\\u00e9\\ud83d\\ude00\"}";
        CHK(lookup("$.s", d2, &sp));
        const std::int32_t w = json_span_unquote(
            sp, buf, static_cast<std::uint32_t>(sizeof(buf)));
        CHK(w > 0);
        const char want[] = "a\"b\\c\nA\xc3\xa9\xf0\x9f\x98\x80";
        CHK(static_cast<std::size_t>(w) == sizeof(want) - 1);
        CHK(std::memcmp(buf, want, sizeof(want) - 1) == 0);
        // non-string values come back verbatim
        JsonSpan num{"1.5", 3};
        CHK(json_span_unquote(num, buf, sizeof(buf)) == 3 &&
            std::memcmp(buf, "1.5", 3) == 0);
        // overflow => -1, never a partial silent write claim
        CHK(json_span_unquote(sp, buf, 2) == -1);
        // bad escape / lone surrogate => -1
        JsonSpan bad{"\"\\q\"", 4};
        CHK(json_span_unquote(bad, buf, sizeof(buf)) == -1);
        JsonSpan lone{"\"\\udc00\"", 8};
        CHK(json_span_unquote(lone, buf, sizeof(buf)) == -1);
    }

    // --- malformed documents: not-found, never a crash -------------------
    {
        JsonSpan sp{};
        const char* bad_docs[] = {
            "not json at all",            // bare garbage scalar w/ spaces
            "{\"a\": ",                   // truncated value
            "{\"a\" 1}",                  // missing colon
            "{\"a\": \"unterminated",     // unterminated string
            "{\"b\": 1,}",                // trailing comma before } (walked)
            "[1, 2",                      // unterminated array
            "{\"a\": 1} trailing",        // trailing garbage
            "",                           // empty
        };
        for (const char* b : bad_docs) {
            CHK(!lookup("$.a", b, &sp));
        }
        // a plain non-JSON string IS a valid scalar doc; keying it fails
        CHK(!lookup("$.a", "\"hello\"", &sp));
        CHK(lookup("$", "\"hello\"", &sp) && span_eq(sp, "\"hello\""));
    }

    // --- skip-nesting depth cap: >32 levels fails cleanly ----------------
    {
        char deep[256];
        std::uint32_t o = 0;
        for (int k = 0; k < 40; ++k) deep[o++] = '[';
        deep[o++] = '1';
        for (int k = 0; k < 40; ++k) deep[o++] = ']';
        deep[o] = '\0';
        JsonSpan sp{};
        CHK(!lookup("$[0]", deep, &sp));      // depth 40 > kMaxSkipDepth
    }

    std::printf("OK bolt_json_path: compile grammar + nested/array lookups + "
                "typed accessors + escape decode + malformed/caps all "
                "value-exact\n");
    return 0;
}
