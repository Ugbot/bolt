// test_bolt_text_fn.cpp — pad / initcap / format_string kernels (Spark semantics, G2LAUNCH-73)
// and one-value nested navigation/building (bolt_nested_value.h, G2LAUNCH-78).

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "bolt/bolt_nested_value.h"
#include "bolt/kernels/bolt_text_fn.h"

namespace tx = bolt::kernels::text;
namespace nr = bolt::nested_row;

namespace {
std::string pad(const std::string& s, int64_t n, const std::string& p, bool left) {
    char out[256];
    const int64_t w = tx::utf8_pad(s.data(), static_cast<uint32_t>(s.size()), n, p.data(),
                                   static_cast<uint32_t>(p.size()), left, out, sizeof(out));
    return w < 0 ? "<err>" : std::string(out, static_cast<size_t>(w));
}
std::string fmt(const char* f, std::initializer_list<tx::FormatArg> args) {
    char out[256];
    const int64_t w = tx::text_format(f, static_cast<uint32_t>(std::strlen(f)), args.begin(),
                                      static_cast<uint32_t>(args.size()), out, sizeof(out));
    return w < 0 ? "<" + std::to_string(w) + ">" : std::string(out, static_cast<size_t>(w));
}
tx::FormatArg S(const char* s) { return {tx::FormatKind::Str, 0, 0, s, static_cast<uint32_t>(std::strlen(s))}; }
tx::FormatArg I(int64_t i) { return {tx::FormatKind::Int, i, 0, nullptr, 0}; }
tx::FormatArg D(double d) { return {tx::FormatKind::Float, 0, d, nullptr, 0}; }
tx::FormatArg N() { return {tx::FormatKind::Null, 0, 0, nullptr, 0}; }
}  // namespace

TEST(BoltTextFn, PadMatchesSpark) {
    EXPECT_EQ(pad("  x ", 5, "*", true), "*  x ");
    EXPECT_EQ(pad("y", 5, "*", true), "****y");
    EXPECT_EQ(pad("hi", 5, "ab", false), "hiaba");
    EXPECT_EQ(pad("hello", 3, "*", true), "hel");
    EXPECT_EQ(pad("hello", 3, "*", false), "hel");
    EXPECT_EQ(pad("hi", 5, "", true), "hi");
    EXPECT_EQ(pad("hi", 0, "*", true), "");
    EXPECT_EQ(pad("\xc3\xa9", 3, "\xc3\xbc", true), "\xc3\xbc\xc3\xbc\xc3\xa9");
}

TEST(BoltTextFn, InitcapLowersThenTitles) {
    const char* in = "aLICE bob  CAROL-dave";
    char out[64];
    const uint32_t n = tx::utf8_initcap(in, static_cast<uint32_t>(std::strlen(in)), out);
    EXPECT_EQ(std::string(out, n), "Alice Bob  Carol-dave");
}

TEST(BoltTextFn, FormatJavaSubset) {
    EXPECT_EQ(fmt("%s-%d", {S("alice"), I(34)}), "alice-34");
    EXPECT_EQ(fmt("%5d|%-5d|%05d", {I(42), I(42), I(-42)}), "   42|42   |-0042");
    EXPECT_EQ(fmt("%.2f %e", {D(3.14159), D(1234.5)}), "3.14 1.234500e+03");
    EXPECT_EQ(fmt("%2$s %1$s %%", {S("a"), S("b")}), "b a %");
    EXPECT_EQ(fmt("%s %d", {N(), N()}), "null null");
    EXPECT_EQ(fmt("%,d %x %S", {I(1234567), I(255), S("ab")}), "1,234,567 ff AB");
    EXPECT_EQ(fmt("%d", {S("x")}), "<-3>");
    EXPECT_EQ(fmt("%s %s", {S("x")}), "<-2>");
    EXPECT_EQ(fmt("%q", {S("x")}), "<-2>");
}

TEST(BoltNestedValue, BuildAndNavigateListOfStrings) {
    nr::TypeTree t{};
    t.n = 2;
    t.nodes[0] = nr::TypeNode{bolt::BoltType::List, bolt::BoltLogical::None, 1, 0, 1, 1, 0, 0, 0};
    t.nodes[1] = nr::TypeNode{bolt::BoltType::Utf8, bolt::BoltLogical::None, 1, 0, 0, 0, 0, 0, 0};
    uint8_t buf[128];
    nr::Writer w{buf, sizeof(buf), 0};
    nr::put_present(&w);
    const size_t at = nr::put_count(&w);
    nr::put_present(&w); nr::put_var(&w, "2021", 4);
    nr::put_null(&w);
    nr::put_present(&w); nr::put_var(&w, "15", 2);
    nr::patch_count(&w, at, 3);
    ASSERT_FALSE(w.failed);
    nr::Span all{};
    ASSERT_TRUE(nr::value_span(t, 0, buf, static_cast<uint32_t>(w.n), &all));
    EXPECT_EQ(all.len, w.n);
    nr::ListCursor c{};
    ASSERT_EQ(nr::list_open(t, 0, all, &c), 1);
    EXPECT_EQ(c.count, 3u);
    nr::Span e{};
    nr::LeafView lv{};
    ASSERT_TRUE(nr::list_next(&c, &e));
    ASSERT_TRUE(nr::leaf_view(t, 1, e, &lv));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(lv.bytes), lv.len), "2021");
    ASSERT_TRUE(nr::list_next(&c, &e));
    ASSERT_TRUE(nr::leaf_view(t, 1, e, &lv));
    EXPECT_TRUE(lv.is_null);
    ASSERT_TRUE(nr::list_next(&c, &e));
    EXPECT_FALSE(nr::list_next(&c, &e));
    // The same bytes decode to a columnar List through the existing codec.
    bolt::Arena arena;
    nr::Decoder d{};
    ASSERT_TRUE(nr::decoder_init(&d, &t, &arena));
    ASSERT_TRUE(nr::decoder_push(&d, buf, w.n));
    bolt::BoltColumn col{};
    ASSERT_TRUE(nr::decoder_finish(d, &col));
    EXPECT_EQ(col.type, bolt::BoltType::List);
    EXPECT_EQ(col.length, 1);
}

TEST(BoltNestedValue, StructFieldAndCompare) {
    nr::TypeTree t{};
    t.n = 3;
    t.nodes[0] = nr::TypeNode{bolt::BoltType::Struct, bolt::BoltLogical::None, 1, 0, 1, 2, 0, 0, 0};
    t.nodes[1] = nr::TypeNode{bolt::BoltType::Utf8, bolt::BoltLogical::None, 1, 0, 0, 0, 0, 0, 0};
    t.nodes[2] = nr::TypeNode{bolt::BoltType::Int64, bolt::BoltLogical::None, 1, 0, 0, 0, 8, 0, 0};
    uint8_t buf[64];
    nr::Writer w{buf, sizeof(buf), 0};
    const int64_t floor = 20;
    nr::put_present(&w);
    nr::put_present(&w); nr::put_var(&w, "sales", 5);
    nr::put_present(&w); nr::put_fixed(&w, &floor, 8);
    nr::Span v{buf, static_cast<uint32_t>(w.n)}, f{};
    ASSERT_EQ(nr::struct_field(t, 0, v, 1, &f), 1);
    nr::LeafView lv{};
    ASSERT_TRUE(nr::leaf_view(t, 2, f, &lv));
    int64_t got = 0;
    std::memcpy(&got, lv.bytes, 8);
    EXPECT_EQ(got, 20);
    EXPECT_EQ(nr::struct_field(t, 0, v, 2, &f), -1);
    const int64_t a = 3, b = 28;
    nr::LeafView la{false, reinterpret_cast<const uint8_t*>(&a), 8}, lb{false, reinterpret_cast<const uint8_t*>(&b), 8};
    EXPECT_LT(nr::leaf_compare(t.nodes[2], la, lb), 0);
}
