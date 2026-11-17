// packed_labels.h against an independent model (split, filter, join) over hand cases and seeded sets.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "bolt/kernels/packed_labels.h"

namespace pl = bolt::kernels::packed_labels;
using bolt::StringView;

namespace {

using Pairs = std::vector<std::pair<std::string, std::string>>;

std::string pack(const Pairs& ps) {
    std::string s;
    for (const auto& p : ps) { s += p.first; s += '\x1f'; s += p.second; s += '\x1e'; }
    return s;
}

std::string model_get(const Pairs& ps, const std::string& name) {
    for (const auto& p : ps)
        if (p.first == name) return p.second;
    return "";
}

std::string model_drop(const Pairs& ps, const std::vector<std::string>& names) {
    Pairs keep;
    for (const auto& p : ps) {
        bool cut = false;
        for (const auto& n : names) cut = cut || n == p.first;
        if (!cut) keep.push_back(p);
    }
    return pack(keep);
}

std::string join_names(const std::vector<std::string>& names) {
    std::string s;
    for (std::size_t i = 0; i < names.size(); ++i) { if (i != 0) s += '\x1f'; s += names[i]; }
    return s;
}

// A column of views over one byte pool, as an operator hands a kernel.
struct Col {
    std::string pool;
    std::vector<StringView> v;
    void add(const std::string& s) {
        StringView x;
        std::memset(&x, 0, sizeof(x));
        x.length = static_cast<std::uint32_t>(s.size());
        if (s.size() <= 12) {
            std::memcpy(x.prefix, s.data(), s.size());
        } else {
            std::memcpy(x.prefix, s.data(), 4);
            x.ref.offset = static_cast<std::uint32_t>(pool.size());
            pool += s;
        }
        v.push_back(x);
    }
};

std::string text(const StringView& s, const char* base) {
    return std::string(bolt::kernels::utf8::sv_bytes(s, base), s.length);
}

std::uint64_t next(std::uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s >> 33;
}

}  // namespace

TEST(PackedLabels, FindReadsOnePairAndNeverAPrefixOrAValue) {
    const std::string s = pack({{"a", "1"}, {"ab", "22"}, {"le", "+Inf"}, {"zz", ""}});
    std::uint32_t off = 0, len = 0;
    ASSERT_TRUE(pl::packed_label_find(s.data(), static_cast<std::uint32_t>(s.size()), "ab", 2, &off, &len));
    EXPECT_EQ(s.substr(off, len), "22");
    ASSERT_TRUE(pl::packed_label_find(s.data(), static_cast<std::uint32_t>(s.size()), "zz", 2, &off, &len));
    EXPECT_EQ(len, 0u);
    EXPECT_FALSE(pl::packed_label_find(s.data(), static_cast<std::uint32_t>(s.size()), "b", 1, &off, &len));
    EXPECT_FALSE(pl::packed_label_find(s.data(), static_cast<std::uint32_t>(s.size()), "22", 2, &off, &len));   // a value
    EXPECT_FALSE(pl::packed_label_find(s.data(), static_cast<std::uint32_t>(s.size()), "l", 1, &off, &len));
    EXPECT_FALSE(pl::packed_label_find("", 0, "a", 1, &off, &len));
    const std::string lead = "\x1e" + s;   // a set a pair was cut from
    ASSERT_TRUE(pl::packed_label_find(lead.data(), static_cast<std::uint32_t>(lead.size()), "a", 1, &off, &len));
    EXPECT_EQ(lead.substr(off, len), "1");
    const std::string open = "a\x1f" "1\x1e" "b\x1f" "tail";   // the last pair without its separator
    ASSERT_TRUE(pl::packed_label_find(open.data(), static_cast<std::uint32_t>(open.size()), "b", 1, &off, &len));
    EXPECT_EQ(open.substr(off, len), "tail");
}

TEST(PackedLabels, GetOverAColumnKeepsLongValuesAsViewsOfTheInput) {
    Col c;
    const Pairs a = {{"exporter", "otlp_http/gestalt"}, {"le", "0.5"}, {"span_name", "a span name well past twelve bytes"}};
    const Pairs b = {{"le", "+Inf"}};
    c.add(pack(a));
    c.add(pack(b));
    c.add("");
    c.add("x\x1fy\x1e");   // inline set
    std::vector<StringView> out(c.v.size());
    pl::packed_label_get(c.v.data(), static_cast<std::int64_t>(c.v.size()), c.pool.data(), "span_name", 9, out.data());
    EXPECT_EQ(text(out[0], c.pool.data()), "a span name well past twelve bytes");
    EXPECT_GT(out[0].length, 12u);
    EXPECT_EQ(out[1].length, 0u);
    EXPECT_EQ(out[2].length, 0u);
    EXPECT_EQ(out[3].length, 0u);
    pl::packed_label_get(c.v.data(), static_cast<std::int64_t>(c.v.size()), c.pool.data(), "le", 2, out.data());
    EXPECT_EQ(text(out[0], c.pool.data()), "0.5");
    EXPECT_EQ(text(out[1], c.pool.data()), "+Inf");
    pl::packed_label_get(c.v.data(), static_cast<std::int64_t>(c.v.size()), c.pool.data(), "x", 1, out.data());
    EXPECT_EQ(text(out[3], c.pool.data()), "y");
    pl::packed_label_get(c.v.data(), static_cast<std::int64_t>(c.v.size()), c.pool.data(), "exporter", 8, out.data());
    EXPECT_EQ(text(out[0], c.pool.data()), "otlp_http/gestalt");
}

TEST(PackedLabels, DropCutsTheListedPairsAndRestoresTheStoredForm) {
    const Pairs ps = {{"__name__", "m"}, {"instance", "i-1"}, {"le", "0.1"}, {"job", "api"}};
    const std::string s = pack(ps);
    char buf[256];
    auto drop = [&](const std::string& in, const std::vector<std::string>& names) {
        const std::string nl = join_names(names);
        const std::uint32_t n = pl::packed_labels_drop(in.data(), static_cast<std::uint32_t>(in.size()), nl.data(),
                                                       static_cast<std::uint32_t>(nl.size()), buf);
        return std::string(buf, n);
    };
    EXPECT_EQ(drop(s, {"le"}), model_drop(ps, {"le"}));
    EXPECT_EQ(drop(s, {"__name__", "job"}), model_drop(ps, {"__name__", "job"}));
    EXPECT_EQ(drop(s, {"nope"}), s);
    EXPECT_EQ(drop(s, {}), s);
    EXPECT_EQ(drop(s, {"l", "e", "insta"}), s);                       // prefixes cut nothing
    EXPECT_EQ(drop(s, {"__name__", "instance", "le", "job"}), "");
    EXPECT_EQ(drop("\x1e" + s, {"instance"}), model_drop(ps, {"instance"}));   // no leading separator in the answer
    EXPECT_EQ(drop("", {"a"}), "");
    EXPECT_EQ(drop("a\x1f" "1", {"b"}), "a\x1f" "1\x1e");            // the open last pair gains its separator
}

TEST(PackedLabels, ColumnKernelsEqualTheModelOnSeededSets) {
    std::uint64_t seed = 0x5eedULL;
    const char* const names[] = {"a", "ab", "le", "job", "instance", "span_name", "exporter", "x_y", "k8s_pod"};
    for (int round = 0; round < 50; ++round) {
        Col c;
        std::vector<Pairs> rows;
        for (int r = 0; r < 64; ++r) {
            Pairs ps;
            for (const char* nm : names) {
                if (next(&seed) % 3 == 0) continue;
                std::string v;
                const std::uint64_t len = next(&seed) % 40;   // 0 = an empty value
                for (std::uint64_t k = 0; k < len; ++k) v += static_cast<char>('a' + next(&seed) % 26);
                ps.emplace_back(nm, v);
            }
            rows.push_back(ps);
            c.add(pack(ps));
        }
        const auto n = static_cast<std::int64_t>(c.v.size());
        std::vector<StringView> out(c.v.size());
        for (const char* nm : names) {
            pl::packed_label_get(c.v.data(), n, c.pool.data(), nm, static_cast<std::uint32_t>(std::strlen(nm)), out.data());
            for (std::size_t r = 0; r < rows.size(); ++r) ASSERT_EQ(text(out[r], c.pool.data()), model_get(rows[r], nm)) << nm;
        }
        std::vector<std::string> cut;
        for (const char* nm : names)
            if (next(&seed) % 2 == 0) cut.push_back(nm);
        const std::string nl = join_names(cut);
        const std::int64_t bound = pl::packed_labels_drop_bound(c.v.data(), n);
        std::vector<char> bytes(static_cast<std::size_t>(bound) + 1);
        const std::int64_t used = pl::packed_labels_drop(c.v.data(), n, c.pool.data(), nl.data(),
                                                         static_cast<std::uint32_t>(nl.size()), out.data(), bytes.data(), bound);
        ASSERT_GE(used, 0);
        ASSERT_LE(used, bound);
        for (std::size_t r = 0; r < rows.size(); ++r) ASSERT_EQ(text(out[r], bytes.data()), model_drop(rows[r], cut));
        if (bound > 1) {   // a buffer one byte short of the first row refuses
            const std::int64_t small = static_cast<std::int64_t>(c.v[0].length);
            EXPECT_EQ(pl::packed_labels_drop(c.v.data(), n, c.pool.data(), nl.data(), static_cast<std::uint32_t>(nl.size()),
                                             out.data(), bytes.data(), small), -1);
        }
    }
}
