// test_bolt_kmerge_bytes.cpp — B10: k-way merge and merge join over
// order-preserving byte keys, checked against std::stable_sort / a nested
// loop model on key_encode-shaped composite keys.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "bolt/join/bolt_mergejoin_bytes.h"
#include "bolt/kernels/bolt_kmerge_bytes.h"

using namespace bolt;

namespace {

// MarbleDB key_encode shape: int64 cells sign-flipped big-endian, then an
// optional var-length final cell.
std::string encode(int64_t a, int64_t b, const std::string& tail) {
    std::string k;
    for (int64_t v : {a, b}) {
        const uint64_t u = static_cast<uint64_t>(v) ^ (1ull << 63);
        for (int s = 56; s >= 0; s -= 8) k.push_back(static_cast<char>((u >> s) & 0xFF));
    }
    return k + tail;
}

std::string random_key(std::mt19937_64& g, int mode) {
    const int64_t small[5] = {-3, -1, 0, 1, 2};
    auto pick = [&](void) -> int64_t {
        return (g() % 4 == 0) ? static_cast<int64_t>(g()) : small[g() % 5];
    };
    std::string tail;
    const int tl = static_cast<int>(g() % 4);
    for (int i = 0; i < tl; ++i) {
        const char alphabet[5] = {'\0', '\x01', 'a', 'b', static_cast<char>(0xFF)};
        tail.push_back(alphabet[g() % 5]);
    }
    if (mode == 0) return encode(pick(), pick(), tail);          // composite + text
    if (mode == 1) return encode(pick(), 0, "").substr(0, 8);    // bare int64
    return tail;                                                 // text only (incl. empty)
}

struct KeyRun {
    std::vector<std::string> keys;
    std::vector<uint8_t> bytes;
    std::vector<int32_t> offs;
    KeyBytesColumn col;
    void finish() {
        std::sort(keys.begin(), keys.end());   // std::string order == memcmp + length
        offs.assign(1, 0);
        bytes.clear();
        for (auto& k : keys) {
            bytes.insert(bytes.end(), k.begin(), k.end());
            offs.push_back(static_cast<int32_t>(bytes.size()));
        }
        bytes.push_back(0);
        col = KeyBytesColumn{bytes.data(), offs.data(), static_cast<int64_t>(keys.size()), 0, 0};
    }
};

using Ref = std::tuple<std::string, uint32_t, int64_t>;   // key, slot, row

}  // namespace

TEST(KMergeBytes, CompareMatchesStdString) {
    std::mt19937_64 g(7);
    for (int t = 0; t < 20000; ++t) {
        const std::string a = random_key(g, t % 3), b = random_key(g, (t / 3) % 3);
        const int c = key_bytes_cmp(reinterpret_cast<const uint8_t*>(a.data()),
                                    static_cast<uint32_t>(a.size()),
                                    reinterpret_cast<const uint8_t*>(b.data()),
                                    static_cast<uint32_t>(b.size()));
        const int r = a.compare(b);
        ASSERT_EQ(c < 0, r < 0);
        ASSERT_EQ(c == 0, r == 0);
    }
}

TEST(KMergeBytes, MatchesStableSortModel) {
    std::mt19937_64 g(42);
    for (int trial = 0; trial < 300; ++trial) {
        const uint32_t k = 1 + static_cast<uint32_t>(g() % 12);
        const int mode = trial % 3;
        std::vector<KeyRun> runs(k);
        std::vector<Ref> model;
        for (uint32_t s = 0; s < k; ++s) {
            const int n = static_cast<int>(g() % 40);
            for (int i = 0; i < n; ++i) runs[s].keys.push_back(random_key(g, mode));
            runs[s].finish();
            for (int64_t r = 0; r < static_cast<int64_t>(runs[s].keys.size()); ++r)
                model.emplace_back(runs[s].keys[static_cast<size_t>(r)], s, r);
            ASSERT_TRUE(kmerge_bytes_check_sorted(runs[s].col));
        }
        std::stable_sort(model.begin(), model.end());   // key, then slot, then row

        std::vector<KMergeBytesInput> in(k);
        for (uint32_t s = 0; s < k; ++s) in[s] = KMergeBytesInput{runs[s].col, 0, 100 + s, 0};
        KMergeBytes m;
        ASSERT_TRUE(kmerge_bytes_init(&m, in.data(), k));
        const int64_t cap = 1 + static_cast<int64_t>(g() % 9);
        std::vector<int32_t> id(static_cast<size_t>(cap));
        std::vector<int64_t> row(static_cast<size_t>(cap));
        std::vector<uint8_t> nk(static_cast<size_t>(cap));
        size_t at = 0;
        for (int guard = 0; guard < 100000; ++guard) {
            const int64_t got = kmerge_bytes_next(&m, id.data(), row.data(), nk.data(), cap);
            if (got == 0) break;
            for (int64_t e = 0; e < got; ++e, ++at) {
                ASSERT_LT(at, model.size());
                const auto& [key, slot, r] = model[at];
                ASSERT_EQ(id[static_cast<size_t>(e)], static_cast<int32_t>(100 + slot));
                ASSERT_EQ(row[static_cast<size_t>(e)], r);
                const bool fresh = at == 0 || std::get<0>(model[at - 1]) != key;
                ASSERT_EQ(nk[static_cast<size_t>(e)], fresh ? 1 : 0) << at;
            }
        }
        ASSERT_EQ(at, model.size());
    }
}

TEST(KMergeBytes, FixedStrideRunsAndResume) {
    // Two runs of 8-byte big-endian keys with a shared duplicate; slot order
    // decides the tie and a partially consumed run resumes at `pos`.
    uint8_t a[3][8] = {}, b[3][8] = {};
    const uint64_t av[3] = {1, 5, 9}, bv[3] = {2, 5, 10};
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 8; ++s) {
            a[i][s] = static_cast<uint8_t>(av[i] >> (56 - 8 * s));
            b[i][s] = static_cast<uint8_t>(bv[i] >> (56 - 8 * s));
        }
    KMergeBytesInput in[2] = {
        {KeyBytesColumn{&a[0][0], nullptr, 3, 8, 0}, 1, 0, 0},   // skip key 1
        {KeyBytesColumn{&b[0][0], nullptr, 3, 8, 0}, 0, 1, 0},
    };
    KMergeBytes m;
    ASSERT_TRUE(kmerge_bytes_init(&m, in, 2));
    int32_t id[8];
    int64_t row[8];
    uint8_t nk[8];
    ASSERT_EQ(kmerge_bytes_next(&m, id, row, nk, 8), 5);
    const int32_t eid[5] = {1, 0, 1, 0, 1};
    const int64_t erow[5] = {0, 1, 1, 2, 2};
    const uint8_t enk[5] = {1, 1, 0, 1, 1};
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(id[i], eid[i]);
        EXPECT_EQ(row[i], erow[i]);
        EXPECT_EQ(nk[i], enk[i]);
    }
    EXPECT_EQ(kmerge_bytes_next(&m, id, row, nk, 8), 0);
}

TEST(KMergeBytes, RejectsBadArguments) {
    KMergeBytes m;
    KMergeBytesInput in[kKMergeMaxInputs + 1] = {};
    EXPECT_FALSE(kmerge_bytes_init(&m, in, 0));
    EXPECT_FALSE(kmerge_bytes_init(&m, in, kKMergeMaxInputs + 1));
    in[0].keys.len = 2;
    in[0].pos = 3;
    EXPECT_FALSE(kmerge_bytes_init(&m, in, 1));
}

TEST(KMergeBytes, ColumnAdapters) {
    Arena arena;
    const char* payload = "aabbbc";
    int32_t offs[4] = {0, 2, 5, 6};
    BoltColumn vb = BoltColumn::make_var_binary(const_cast<char*>(payload), nullptr, offs,
                                                3, BoltType::Binary, &arena);
    KeyBytesColumn k;
    ASSERT_TRUE(key_bytes_from_column(vb, &k));
    EXPECT_EQ(k.len, 3);
    const uint8_t* p = nullptr;
    uint32_t n = 0;
    key_bytes_at(k, 1, &p, &n);
    EXPECT_EQ(n, 3u);
    EXPECT_EQ(std::memcmp(p, "bbb", 3), 0);
    BoltColumn fx = BoltColumn::make_flat_alloc(4, BoltType::UUID, &arena);
    ASSERT_TRUE(key_bytes_from_column(fx, &k));
    EXPECT_EQ(k.stride, 16u);
    BoltColumn u8 = BoltColumn::make_flat_alloc(4, BoltType::Utf8, &arena);
    EXPECT_FALSE(key_bytes_from_column(u8, &k));
}

namespace {

using Pair = std::pair<int64_t, int64_t>;

std::vector<Pair> join_model(const KeyRun& b, const KeyRun& p, MergeJoinKind kind) {
    // Key order; within a key: probe-major cross product, unmatched rows in
    // their own key position.
    std::vector<std::tuple<std::string, int, Pair>> out;   // key, seq, pair
    const bool keep_b = kind == MergeJoinKind::RightOuter || kind == MergeJoinKind::FullOuter;
    const bool keep_p = kind == MergeJoinKind::LeftOuter || kind == MergeJoinKind::FullOuter;
    int seq = 0;
    for (size_t j = 0; j < p.keys.size(); ++j) {
        bool hit = false;
        for (size_t i = 0; i < b.keys.size(); ++i)
            if (b.keys[i] == p.keys[j]) {
                out.emplace_back(p.keys[j], seq++, Pair(int64_t(i), int64_t(j)));
                hit = true;
            }
        if (!hit && keep_p) out.emplace_back(p.keys[j], seq++, Pair(-1, int64_t(j)));
    }
    for (size_t i = 0; i < b.keys.size(); ++i) {
        const bool hit = std::find(p.keys.begin(), p.keys.end(), b.keys[i]) != p.keys.end();
        if (!hit && keep_b) out.emplace_back(b.keys[i], seq++, Pair(int64_t(i), -1));
    }
    std::stable_sort(out.begin(), out.end(), [](const auto& x, const auto& y) {
        if (std::get<0>(x) != std::get<0>(y)) return std::get<0>(x) < std::get<0>(y);
        // unmatched build rows of equal keys sort by build index
        const Pair& px = std::get<2>(x);
        const Pair& py = std::get<2>(y);
        if (px.second == -1 && py.second == -1) return px.first < py.first;
        return false;
    });
    std::vector<Pair> r;
    for (auto& t : out) r.push_back(std::get<2>(t));
    return r;
}

}  // namespace

TEST(MergeJoinBytes, MatchesNestedLoopModelAllKinds) {
    std::mt19937_64 g(99);
    for (int trial = 0; trial < 400; ++trial) {
        KeyRun b, p;
        const int mode = trial % 3;
        const int bn = static_cast<int>(g() % 30), pn = static_cast<int>(g() % 30);
        for (int i = 0; i < bn; ++i) b.keys.push_back(random_key(g, mode));
        for (int i = 0; i < pn; ++i) p.keys.push_back(random_key(g, mode));
        b.finish();
        p.finish();
        for (uint8_t kk = 0; kk < 4; ++kk) {
            const MergeJoinKind kind = static_cast<MergeJoinKind>(kk);
            const std::vector<Pair> want = join_model(b, p, kind);
            MergeJoinBytesCursor c;
            ASSERT_TRUE(mergejoin_bytes_init(&c, b.col, p.col, kind));
            const int64_t cap = 1 + static_cast<int64_t>(g() % 7);
            std::vector<int64_t> ob(static_cast<size_t>(cap)), op(static_cast<size_t>(cap));
            std::vector<Pair> got;
            for (int guard = 0; guard < 100000; ++guard) {
                const int64_t n = mergejoin_bytes_next(&c, ob.data(), op.data(), cap);
                if (n == 0) break;
                for (int64_t e = 0; e < n; ++e)
                    got.emplace_back(ob[static_cast<size_t>(e)], op[static_cast<size_t>(e)]);
            }
            ASSERT_EQ(got, want) << "trial " << trial << " kind " << int(kk);
        }
    }
}
