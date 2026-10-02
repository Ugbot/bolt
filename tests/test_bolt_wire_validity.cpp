// test_bolt_wire_validity.cpp — B4: the wire serializers honour
// BoltColumn::validity_offset (a sliced column's nulls start mid-bitmap).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_stream.h"

using namespace bolt;

namespace {

constexpr int64_t kParentRows = 160;

// Deterministic, non-periodic null pattern (period 7 never aligns with bytes).
bool parent_valid(int64_t r) { return (r * 5 + 3) % 7 != 0; }

void set_bit(uint8_t* bm, int64_t i, bool v) {
    if (v) bm[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    else   bm[i >> 3] &= static_cast<uint8_t>(~(1u << (i & 7)));
}

struct Parent {
    int64_t values[kParentRows];
    uint8_t validity[kParentRows / 8 + 1];
};

void build_parent(Parent* p) {
    std::memset(p->validity, 0xFF, sizeof(p->validity));   // garbage past the end
    for (int64_t r = 0; r < kParentRows; ++r) {
        p->values[r] = parent_valid(r) ? r * 11 : 0;
        set_bit(p->validity, r, parent_valid(r));
    }
}

// A Flat Int64 slice [off, off + len) of the parent, the shape a windowed /
// chunked caller hands the serializer: data advanced, validity_offset = off.
BoltColumn slice_of(Parent* p, int64_t off, int64_t len) {
    BoltColumn c = BoltColumn::make_flat(p->values + off, p->validity, len,
                                         BoltType::Int64);
    c.validity_offset = off;
    c.stats.all_valid = false;
    return c;
}

void one_col_batch(BoltBatch* b, Arena* a, const BoltColumn& c) {
    BoltBatch::init_empty(b);
    b->num_rows = c.length;
    b->arena = a;
    ASSERT_TRUE(BoltBatch::alloc_columns(b, a, 1));
    b->schema.num_fields = 1;
    BoltField& f = b->schema.fields[0];
    std::memset(&f, 0, sizeof(f));
    f.set_name("v");
    f.type = c.type;
    f.nullable = true;
    b->columns[0][0] = c;
    b->columns[1][0] = c;
}

std::vector<uint8_t> serialize(const BoltBatch& b) {
    const size_t need = wire::bolt_wire_size(&b);
    EXPECT_GT(need, 0u);
    std::vector<uint8_t> buf(need, 0xCD);
    EXPECT_EQ(wire::bolt_wire_serialize(&b, buf.data(), buf.size()), need);
    return buf;
}

}  // namespace

// Every (offset, length) slice round-trips its nulls and values.
TEST(WireValidity, SlicedColumnRoundTripsEveryOffset) {
    Parent p;
    build_parent(&p);
    for (int64_t off = 0; off < 64; ++off) {
        for (int64_t len = 0; len + off <= kParentRows && len < 72; ++len) {
            Arena a;
            BoltBatch src;
            one_col_batch(&src, &a, slice_of(&p, off, len));
            std::vector<uint8_t> buf = serialize(src);
            BoltBatch dst;
            ASSERT_TRUE(wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &a));
            const BoltColumn& d = dst.col(0);
            ASSERT_EQ(d.length, len);
            for (int64_t r = 0; r < len; ++r) {
                const bool v = parent_valid(off + r);
                ASSERT_EQ(!d.is_null(r), v) << "off=" << off << " len=" << len << " r=" << r;
                if (v) ASSERT_EQ(static_cast<const int64_t*>(d.data)[r], (off + r) * 11);
            }
        }
    }
}

// Bits past the last row are zero on the wire, whatever the source holds there.
TEST(WireValidity, TrailingBitsAreZero) {
    Parent p;
    build_parent(&p);
    for (int64_t off = 0; off < 16; ++off) {
        const int64_t len = 13;
        Arena a;
        BoltBatch src;
        one_col_batch(&src, &a, slice_of(&p, off, len));
        std::vector<uint8_t> buf = serialize(src);
        uint64_t o0 = 0, l0 = 0;
        const uint32_t data_off = wire::detail::read_u32_le(buf.data() + 28);
        (void)data_off;
        const size_t desc = wire::kWireHeaderSize + wire::kWireSchemaEntrySize;
        o0 = wire::detail::read_u64_le(buf.data() + desc + 0);
        l0 = wire::detail::read_u64_le(buf.data() + desc + 8);
        ASSERT_EQ(l0, 2u);
        EXPECT_EQ(buf[o0 + 1] & 0xE0u, 0u) << "off=" << off;
    }
}

// Two slices with identical logical content serialize to identical bytes.
TEST(WireValidity, SameLogicalSliceSameBytes) {
    Parent p;
    build_parent(&p);
    // Rows [7, 7+20) and [14, 14+20) share the null pattern (period 7) and
    // differ only in values; rebuild a twin parent so values match too.
    Parent q;
    build_parent(&q);
    for (int64_t r = 0; r + 7 < kParentRows; ++r) q.values[r + 7] = p.values[r];
    Arena a, b;
    BoltBatch s1, s2;
    one_col_batch(&s1, &a, slice_of(&p, 7, 20));
    one_col_batch(&s2, &b, slice_of(&q, 14, 20));
    EXPECT_EQ(serialize(s1), serialize(s2));
}

// A Flat Utf8 slice with nulls and spilled rows (exercises the path that
// already honoured validity_offset for sizing, and now for the bitmap).
TEST(WireValidity, Utf8SliceNullsRoundTrip) {
    Arena a;
    const int64_t n = 40;
    const char* words[4] = {"a", "bbbbbbbbbbbbbbbbbbbbbbb", "cc", "dddddddddddddddddddd"};
    uint8_t* bm = static_cast<uint8_t*>(a.allocate(8, 64));
    std::memset(bm, 0, 8);
    BoltColumn parent = BoltColumn::make_flat_alloc(n, BoltType::Utf8, &a);
    auto* sv = static_cast<StringView*>(parent.data);
    char* pool = static_cast<char*>(a.allocate(1024, 64));
    uint32_t used = 0;
    for (int64_t r = 0; r < n; ++r) {
        const char* w = words[r % 4];
        const uint32_t len = static_cast<uint32_t>(std::strlen(w));
        std::memcpy(pool + used, w, len);
        sv[r] = StringView::from_cstr(w);
        if (len > 12) { sv[r].ref.buf_idx = 0; sv[r].ref.offset = used; }
        used += len;
        set_bit(bm, r, parent_valid(r));
    }
    parent.str_overflow_base = pool;
    for (int64_t off : {1, 3, 9}) {
        BoltColumn c = parent;
        c.data = sv + off;
        c.validity = bm;
        c.validity_offset = off;
        c.length = 17;
        c.stats.all_valid = false;
        BoltBatch src;
        one_col_batch(&src, &a, c);
        std::vector<uint8_t> buf = serialize(src);
        BoltBatch dst;
        ASSERT_TRUE(wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &a));
        const BoltColumn& d = dst.col(0);
        for (int64_t r = 0; r < c.length; ++r) {
            ASSERT_EQ(!d.is_null(r), parent_valid(off + r)) << off << "/" << r;
            if (!parent_valid(off + r)) continue;
            const uint8_t* data = nullptr;
            int32_t len = 0;
            d.utf8_at(r, &data, &len);
            const char* w = words[(off + r) % 4];
            ASSERT_EQ(len, static_cast<int32_t>(std::strlen(w)));
            ASSERT_EQ(std::memcmp(data, w, static_cast<size_t>(len)), 0);
        }
    }
}

// The streaming writer copies the bitmap the same way.
TEST(WireValidity, StreamWriterHonoursOffset) {
    Parent p;
    build_parent(&p);
    for (int64_t off : {0, 1, 5, 8, 13}) {
        const int64_t len = 29;
        BoltColumn c = slice_of(&p, off, len);
        Arena a;
        BoltBatch src;
        one_col_batch(&src, &a, c);
        std::vector<uint8_t> ref = serialize(src);

        std::vector<uint8_t> out(ref.size() + 4096, 0);
        WireStream s;
        BoltField f = src.schema.fields[0];
        ASSERT_TRUE(wire_stream_begin_file(&s, out.data(), out.size(), &f, 1));
        ASSERT_TRUE(wire_stream_append_column(&s, &c, c.data,
                                              static_cast<size_t>(len) * 8));
        const size_t n = wire_stream_finalize(&s);
        ASSERT_GT(n, 0u);
        BoltBatch dst;
        ASSERT_TRUE(wire::bolt_wire_deserialize(out.data(), n, &dst, &a));
        for (int64_t r = 0; r < len; ++r) {
            ASSERT_EQ(!dst.col(0).is_null(r), parent_valid(off + r)) << off << "/" << r;
        }
    }
}

// A whole column (offset 0) whose bitmap tail is already clean keeps its
// exact pre-B4 bytes: frames written before the fix stay byte-identical.
TEST(WireValidity, OffsetZeroCleanTailUnchanged) {
    Parent p;
    build_parent(&p);
    const int64_t len = 64;   // whole bytes: no tail bits at all
    Arena a;
    BoltBatch src;
    one_col_batch(&src, &a, slice_of(&p, 0, len));
    std::vector<uint8_t> buf = serialize(src);
    const size_t desc = wire::kWireHeaderSize + wire::kWireSchemaEntrySize;
    const uint64_t o0 = wire::detail::read_u64_le(buf.data() + desc + 0);
    EXPECT_EQ(std::memcmp(buf.data() + o0, p.validity, 8), 0);
}

// 20 seeds: random validity patterns, offsets and lengths over an Int64 and
// a Flat Utf8 column sliced at the same offset.
TEST(WireValidity, SeededFuzzRandomSlices) {
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        std::mt19937_64 g(seed);
        Arena a;
        const int64_t n = 64 + static_cast<int64_t>(g() % 400);
        auto* bm = static_cast<uint8_t*>(a.allocate(static_cast<size_t>(n / 8 + 1), 64));
        auto* vals = static_cast<int64_t*>(a.allocate(static_cast<size_t>(n) * 8, 64));
        auto* sv = static_cast<StringView*>(a.allocate(static_cast<size_t>(n) * 16, 64));
        char* pool = static_cast<char*>(a.allocate(static_cast<size_t>(n) * 32, 64));
        uint32_t used = 0;
        for (int64_t r = 0; r < n / 8 + 1; ++r) bm[r] = static_cast<uint8_t>(g());
        for (int64_t r = 0; r < n; ++r) {
            vals[r] = static_cast<int64_t>(g());
            char t[32];
            const int len = std::snprintf(t, sizeof(t), "%s%llu", (g() & 1) ? "spilled-string-" : "",
                                          static_cast<unsigned long long>(r));
            sv[r] = StringView::from_cstr(t);
            if (len > 12) {
                std::memcpy(pool + used, t, static_cast<size_t>(len));
                sv[r].ref.buf_idx = 0;
                sv[r].ref.offset = used;
                used += static_cast<uint32_t>(len);
            }
        }
        for (int iter = 0; iter < 50; ++iter) {
            const int64_t off = static_cast<int64_t>(g() % static_cast<uint64_t>(n));
            const int64_t len = static_cast<int64_t>(g() % static_cast<uint64_t>(n - off + 1));
            BoltColumn ci = BoltColumn::make_flat(vals + off, bm, len, BoltType::Int64);
            ci.validity_offset = off;
            BoltColumn cs = BoltColumn::make_flat(sv + off, bm, len, BoltType::Utf8);
            cs.validity_offset = off;
            cs.str_overflow_base = pool;
            for (const BoltColumn* c : {&ci, &cs}) {
                BoltBatch src;
                one_col_batch(&src, &a, *c);
                std::vector<uint8_t> buf = serialize(src);
                BoltBatch dst;
                ASSERT_TRUE(wire::bolt_wire_deserialize(buf.data(), buf.size(), &dst, &a));
                for (int64_t r = 0; r < len; ++r) {
                    const int64_t b = off + r;
                    const bool valid = (bm[b >> 3] >> (b & 7)) & 1u;
                    ASSERT_EQ(!dst.col(0).is_null(r), valid)
                        << "seed " << seed << " off " << off << " r " << r;
                    if (!valid) continue;
                    const uint8_t *p1, *p2;
                    int32_t n1, n2;
                    if (c->type == BoltType::Int64) {
                        ASSERT_EQ(static_cast<const int64_t*>(dst.col(0).data)[r], vals[b]);
                    } else {
                        c->utf8_at(r, &p1, &n1);
                        dst.col(0).utf8_at(r, &p2, &n2);
                        ASSERT_EQ(n1, n2);
                        ASSERT_EQ(std::memcmp(p1, p2, static_cast<size_t>(n1)), 0);
                    }
                }
            }
        }
    }
}
