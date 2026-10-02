// test_bolt_wire_kinds.cpp — B3: wire ids for the hot column kinds
// (Constant, Sequence, Dictionary, RLE, Nested List/Map/Struct), the missing
// fixed-width types, Flat Binary/Symbol, BoltLogical, version stamping, and
// zero-copy views over a 64 B-aligned buffer.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_frame_file.h"

using namespace bolt;
using namespace bolt::wire;

namespace {

struct Buf {
    std::vector<uint8_t> raw;
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit Buf(size_t bytes) : raw(bytes + 64, 0xEE) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(raw.data());
        p = raw.data() + ((64 - (a & 63)) & 63);
        n = bytes;
    }
    bool holds(const void* q) const {
        const uint8_t* b = static_cast<const uint8_t*>(q);
        return b >= p && b < p + n;
    }
};

std::string sv_bytes(const StringView& sv, const void* overflow) {
    if (sv.length <= 12) return std::string(sv.prefix, sv.length);
    return std::string(static_cast<const char*>(overflow) + sv.ref.offset, sv.length);
}

// Row r's value as bytes, whatever the format; "<null>" for a null row.
std::string value_at(const BoltColumn& c, int64_t r) {
    if (c.is_null(r)) return "<null>";
    const bool str = c.type == BoltType::Utf8 || c.type == BoltType::Binary ||
                     c.type == BoltType::Symbol;
    const size_t tsz = str ? 16 : type_size(c.type);
    switch (c.format) {
        case ColumnFormat::Flat:
        case ColumnFormat::View: {
            if (str) return sv_bytes(static_cast<const StringView*>(c.data)[r], c.str_overflow_base);
            return std::string(static_cast<const char*>(c.data) + r * c.type_size_bytes,
                               c.type_size_bytes);
        }
        case ColumnFormat::Constant: {
            if (str) {
                StringView sv; std::memcpy(&sv, c.inline_value, 16);
                return sv_bytes(sv, c.str_overflow_base);
            }
            return std::string(reinterpret_cast<const char*>(c.inline_value), tsz);
        }
        case ColumnFormat::Sequence: {
            const int64_t v = c.seq_offset + r * c.seq_step;
            return std::string(reinterpret_cast<const char*>(&v), tsz);
        }
        case ColumnFormat::RLE: {
            const int32_t* ends = static_cast<const int32_t*>(c.dict_child->data);
            const int64_t i = std::upper_bound(ends, ends + c.dict_child->length,
                                               static_cast<int32_t>(r)) - ends;
            return std::string(static_cast<const char*>(c.data) + static_cast<size_t>(i) * tsz, tsz);
        }
        case ColumnFormat::Dictionary: {
            uint32_t k = 0;
            std::memcpy(&k, static_cast<const uint8_t*>(c.data) + r * c.type_size_bytes,
                        c.type_size_bytes);
            return value_at(*c.dict_child, k);
        }
        case ColumnFormat::VarBinary: {
            const uint8_t* p; int32_t n;
            c.var_binary_at(r, &p, &n);
            return std::string(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
        }
        case ColumnFormat::Nested: {
            const BoltColumn* kids = static_cast<const BoltColumn*>(c.data);
            std::string s = "{";
            if (c.type == BoltType::Struct) {
                for (int64_t k = 0; k < c.seq_offset; ++k) s += value_at(kids[k], r) + "|";
            } else {
                const int32_t* o = static_cast<const int32_t*>(c.dict_child->data);
                for (int32_t e = o[r]; e < o[r + 1]; ++e) s += value_at(kids[0], e) + ",";
            }
            return s + "}";
        }
        default: return "?";
    }
}

struct Kit {
    Arena a;
    std::vector<BoltColumn> cols;
    std::vector<BoltField> fields;
    int64_t n = 0;

    void add(const char* name, BoltColumn c) {
        BoltField f;
        std::memset(&f, 0, sizeof(f));
        f.set_name(name);
        f.type = c.type;
        f.nullable = c.validity != nullptr;
        if (c.type == BoltType::FixedSizeBinary) f.fixed_size = c.fixed_width;
        cols.push_back(c);
        fields.push_back(f);
    }
    void batch(BoltBatch* b) {
        BoltBatch::init_empty(b);
        ASSERT_TRUE(BoltBatch::alloc_columns(b, &a, static_cast<uint32_t>(cols.size())));
        b->num_rows = n;
        b->schema.num_fields = static_cast<uint32_t>(cols.size());
        for (size_t i = 0; i < cols.size(); ++i) {
            b->schema.fields[i] = fields[i];
            b->columns[0][i] = cols[i];
            b->columns[1][i] = cols[i];
        }
    }
    uint8_t* bitmap(uint64_t pattern) {
        auto* v = static_cast<uint8_t*>(a.allocate(static_cast<size_t>((n + 7) / 8) + 8, 64));
        std::memset(v, 0, static_cast<size_t>((n + 7) / 8) + 8);
        for (int64_t r = 0; r < n; ++r)
            if ((pattern >> (r % 64)) & 1u) v[r >> 3] |= uint8_t(1u << (r & 7));
        return v;
    }
    StringView str(const std::string& s, char* pool, uint32_t* used) {
        StringView sv;
        std::memset(&sv, 0, sizeof(sv));
        sv.length = static_cast<uint32_t>(s.size());
        std::memcpy(sv.prefix, s.data(), s.size() < 4 ? s.size() : 4);
        if (s.size() <= 12) {
            if (s.size() > 4) std::memcpy(sv.inline_data, s.data() + 4, s.size() - 4);
        } else {
            std::memcpy(pool + *used, s.data(), s.size());
            sv.ref.offset = *used;
            *used += static_cast<uint32_t>(s.size());
        }
        return sv;
    }
};

void build_all_kinds(Kit* k) {
    const int64_t n = k->n = 50;
    Arena& a = k->a;
    char* pool = static_cast<char*>(a.allocate(8192, 64));
    uint32_t used = 0;

    k->add("const_i64", BoltColumn::make_constant<int64_t>(-42, n, BoltType::Int64));
    (void)k->str("filler so the constant does not spill at offset 0", pool, &used);
    StringView long_sv = k->str("a constant longer than twelve", pool, &used);
    BoltColumn cs = BoltColumn::make_constant(long_sv, n, BoltType::Utf8);
    cs.str_overflow_base = pool;
    k->add("const_str", cs);
    BoltColumn cn = BoltColumn::make_constant<int32_t>(0, n, BoltType::Int32);
    cn.validity = k->bitmap(0);
    cn.stats.all_valid = false;
    k->add("const_null", cn);
    k->add("seq", BoltColumn::make_sequence(1000, -7, n, BoltType::Int64));

    // RLE int32: runs of 7.
    const int64_t runs = (n + 6) / 7;
    auto* rv = a.allocate_array<int32_t>(static_cast<size_t>(runs));
    auto* re = a.allocate_array<int32_t>(static_cast<size_t>(runs));
    for (int64_t i = 0; i < runs; ++i) {
        rv[i] = static_cast<int32_t>(i * i - 3);
        re[i] = static_cast<int32_t>(std::min<int64_t>((i + 1) * 7, n));
    }
    k->add("rle", BoltColumn::make_rle(rv, runs, re, n, BoltType::Int32, &a));

    // Dictionary: u16 keys into 5 Utf8 values (two spilled), with nulls.
    BoltColumn* dv = a.allocate_array<BoltColumn>(1);
    *dv = BoltColumn::make_flat_alloc(5, BoltType::Utf8, &a);
    const char* words[5] = {"x", "yy", "a long dictionary value", "zzz", "another long one!!"};
    for (int i = 0; i < 5; ++i) static_cast<StringView*>(dv->data)[i] = k->str(words[i], pool, &used);
    dv->str_overflow_base = pool;
    BoltColumn dc = BoltColumn::make_empty();
    dc.format = ColumnFormat::Dictionary;
    dc.type = BoltType::Utf8;
    dc.length = n;
    dc.type_size_bytes = 2;
    auto* keys = a.allocate_array<uint16_t>(static_cast<size_t>(n));
    for (int64_t r = 0; r < n; ++r) keys[r] = static_cast<uint16_t>((r * 3) % 5);
    dc.data = keys;
    dc.dict_child = dv;
    dc.validity = k->bitmap(0xF0F0F0F0F0F0F0F0ull);
    dc.stats.all_valid = false;
    k->add("dict", dc);

    // List<Int64> with empty and null lists.
    auto* offs = a.allocate_array<int32_t>(static_cast<size_t>(n + 1));
    offs[0] = 0;
    for (int64_t r = 0; r < n; ++r) offs[r + 1] = offs[r] + static_cast<int32_t>(r % 4);
    BoltColumn elem = BoltColumn::make_flat_alloc(offs[n], BoltType::Int64, &a);
    for (int32_t e = 0; e < offs[n]; ++e) static_cast<int64_t*>(elem.data)[e] = e * 10;
    k->add("list", BoltColumn::make_list(&elem, offs, n, k->bitmap(0x7FFFFFFFFFFFFFFEull), &a));

    // Struct {Int32 with nulls, Utf8, List<Int64>}.
    BoltColumn sf[3];
    sf[0] = BoltColumn::make_flat_alloc(n, BoltType::Int32, &a);
    for (int64_t r = 0; r < n; ++r) static_cast<int32_t*>(sf[0].data)[r] = static_cast<int32_t>(r * 3);
    sf[0].validity = k->bitmap(0x5555555555555555ull);
    sf[0].stats.all_valid = false;
    sf[1] = BoltColumn::make_flat_alloc(n, BoltType::Utf8, &a);
    for (int64_t r = 0; r < n; ++r)
        static_cast<StringView*>(sf[1].data)[r] =
            k->str(r % 2 ? "short" : "struct field long string", pool, &used);
    sf[1].str_overflow_base = pool;
    sf[2] = k->cols.back();
    k->add("struct", BoltColumn::make_struct(sf, 3, n, nullptr, &a));

    // Map<Int32, Int64> = List<Struct{key, value}>.
    BoltColumn kv[2];
    kv[0] = BoltColumn::make_flat_alloc(offs[n], BoltType::Int32, &a);
    kv[1] = BoltColumn::make_flat_alloc(offs[n], BoltType::Int64, &a);
    for (int32_t e = 0; e < offs[n]; ++e) {
        static_cast<int32_t*>(kv[0].data)[e] = e;
        static_cast<int64_t*>(kv[1].data)[e] = -e;
    }
    BoltColumn entries = BoltColumn::make_struct(kv, 2, offs[n], nullptr, &a);
    BoltColumn map = BoltColumn::make_list(&entries, offs, n, nullptr, &a);
    map.type = BoltType::Map;
    k->add("map", map);

    // Missing fixed-width types, Flat Binary / Symbol, a JSON logical tag.
    const BoltType fixed[7] = {BoltType::Date64, BoltType::Duration, BoltType::UUID,
                               BoltType::IPv4, BoltType::FixedSizeBinary,
                               BoltType::Decimal256, BoltType::Float16};
    for (BoltType t : fixed) {
        BoltColumn c = BoltColumn::make_flat_alloc(n, t, &a);
        auto* d = static_cast<uint8_t*>(c.data);
        for (size_t i = 0; i < static_cast<size_t>(n) * c.type_size_bytes; ++i)
            d[i] = static_cast<uint8_t>(i * 13 + static_cast<size_t>(t));
        if (t == BoltType::FixedSizeBinary) c.fixed_width = 12;
        if (t == BoltType::Decimal256) c.decimal_scale = 9;
        k->add(type_name(t), c);
    }
    for (BoltType t : {BoltType::Binary, BoltType::Symbol}) {
        BoltColumn c = BoltColumn::make_flat_alloc(n, BoltType::Utf8, &a);
        c.type = t;
        for (int64_t r = 0; r < n; ++r)
            static_cast<StringView*>(c.data)[r] =
                k->str(std::string(static_cast<size_t>(r % 20), char('A' + r % 26)), pool, &used);
        c.str_overflow_base = pool;
        k->add(t == BoltType::Binary ? "bin" : "sym", c);
    }
    BoltColumn js = BoltColumn::make_flat_alloc(n, BoltType::Utf8, &a);
    for (int64_t r = 0; r < n; ++r) static_cast<StringView*>(js.data)[r] = k->str("{}", pool, &used);
    js.logical = BoltLogical::Json;
    k->add("json", js);
}

void expect_same(const BoltBatch& src, const BoltBatch& dst) {
    ASSERT_EQ(dst.num_cols, src.num_cols);
    ASSERT_EQ(dst.num_rows, src.num_rows);
    for (uint32_t c = 0; c < src.num_cols; ++c) {
        const BoltColumn& s = src.col(c);
        const BoltColumn& d = dst.col(c);
        ASSERT_EQ(d.format == ColumnFormat::Flat && s.format == ColumnFormat::View
                      ? ColumnFormat::View : d.format, s.format) << c;
        ASSERT_EQ(d.type, s.type) << c;
        ASSERT_EQ(d.logical, s.logical) << c;
        ASSERT_EQ(d.decimal_scale, s.decimal_scale) << c;
        ASSERT_EQ(std::string(dst.schema.fields[c].name), std::string(src.schema.fields[c].name));
        for (int64_t r = 0; r < src.num_rows; ++r)
            ASSERT_EQ(value_at(d, r), value_at(s, r)) << src.schema.fields[c].name << " row " << r;
    }
}

uint32_t header_version(const uint8_t* p) {
    uint32_t v; std::memcpy(&v, p + 4, 4); return v;
}

}  // namespace

TEST(WireKinds, EveryKindRoundTripsDeserializeAndView) {
    Kit k;
    build_all_kinds(&k);
    BoltBatch src;
    k.batch(&src);
    const size_t need = bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    Buf buf(need);
    ASSERT_EQ(bolt_wire_serialize(&src, buf.p, buf.n), need);
    EXPECT_EQ(header_version(buf.p), kWireVersionEncoded);

    Arena out;
    BoltBatch copy, view;
    ASSERT_TRUE(bolt_wire_deserialize(buf.p, need, &copy, &out));
    expect_same(src, copy);
    ASSERT_TRUE(bolt_wire_view(buf.p, need, &view, &out));
    expect_same(src, view);

    // Zero copy: every value buffer of the view aliases the blob, 64 B aligned.
    for (uint32_t c = 0; c < view.num_cols; ++c) {
        const BoltColumn& v = view.col(c);
        if (v.format == ColumnFormat::Constant || v.format == ColumnFormat::Sequence) continue;
        if (v.format == ColumnFormat::Nested) {
            const BoltColumn* kid = static_cast<const BoltColumn*>(v.data);
            if (kid[0].format == ColumnFormat::Flat)
                EXPECT_TRUE(buf.holds(kid[0].data)) << c;
            if (v.dict_child) EXPECT_TRUE(buf.holds(v.dict_child->data)) << c;
            continue;
        }
        EXPECT_TRUE(buf.holds(v.data)) << view.schema.fields[c].name;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(v.data) % 64, 0u) << view.schema.fields[c].name;
        if (v.format == ColumnFormat::Dictionary || v.format == ColumnFormat::RLE)
            EXPECT_TRUE(buf.holds(v.dict_child->data)) << c;
    }
    // The deserialized copy owns its bytes.
    for (uint32_t c = 0; c < copy.num_cols; ++c)
        if (copy.col(c).format == ColumnFormat::Flat) EXPECT_FALSE(buf.holds(copy.col(c).data));
}

TEST(WireKinds, PlainBatchesStayVersion4) {
    Arena a;
    BoltBatch b;
    BoltBatch::init_empty(&b);
    ASSERT_TRUE(BoltBatch::alloc_columns(&b, &a, 2));
    b.num_rows = 8;
    b.schema.num_fields = 2;
    const BoltType ts[2] = {BoltType::Int64, BoltType::Float16};
    for (uint32_t i = 0; i < 2; ++i) {
        std::memset(&b.schema.fields[i], 0, sizeof(BoltField));
        b.schema.fields[i].type = ts[i];
        BoltColumn c = BoltColumn::make_flat_alloc(8, ts[i], &a);
        std::memset(c.data, 1, 8 * c.type_size_bytes);
        b.columns[0][i] = b.columns[1][i] = c;
    }
    std::vector<uint8_t> buf(bolt_wire_size(&b));
    ASSERT_EQ(bolt_wire_serialize(&b, buf.data(), buf.size()), buf.size());
    EXPECT_EQ(header_version(buf.data()), 4u);
    // The descriptor tail stays zero for v4 columns.
    const uint8_t* d = buf.data() + kWireHeaderSize + 2 * kWireSchemaEntrySize;
    for (size_t i = 49; i < 56; ++i) EXPECT_EQ(d[i], 0u);
}

TEST(WireKinds, NewPairUnderOldVersionIsRefused) {
    Kit k;
    k.n = 4;
    k.add("seq", BoltColumn::make_sequence(1, 1, 4, BoltType::Int64));
    BoltBatch src;
    k.batch(&src);
    std::vector<uint8_t> buf(bolt_wire_size(&src));
    ASSERT_EQ(bolt_wire_serialize(&src, buf.data(), buf.size()), buf.size());
    Arena a;
    BoltBatch dst;
    ASSERT_TRUE(bolt_wire_deserialize(buf.data(), buf.size(), &dst, &a));
    const uint32_t v4 = 4;
    std::memcpy(buf.data() + 4, &v4, 4);       // what a v4 reader would require
    EXPECT_FALSE(bolt_wire_deserialize(buf.data(), buf.size(), &dst, &a));
}

TEST(WireKinds, CorruptEncodedLayoutsAreRefused) {
    Kit k;
    build_all_kinds(&k);
    BoltBatch src;
    k.batch(&src);
    std::vector<uint8_t> good(bolt_wire_size(&src));
    ASSERT_EQ(bolt_wire_serialize(&src, good.data(), good.size()), good.size());
    const size_t desc0 = kWireHeaderSize + src.num_cols * kWireSchemaEntrySize;
    auto desc = [&](uint32_t c) { return desc0 + c * kWireDescSize; };
    Arena a;
    BoltBatch dst;
    // RLE (col 4): last run end != rows.
    std::vector<uint8_t> bad = good;
    uint64_t o2; std::memcpy(&o2, bad.data() + desc(4) + 32, 8);
    uint64_t l2; std::memcpy(&l2, bad.data() + desc(4) + 40, 8);
    int32_t wrong = 49;
    std::memcpy(bad.data() + o2 + l2 - 4, &wrong, 4);
    EXPECT_FALSE(bolt_wire_deserialize(bad.data(), bad.size(), &dst, &a));
    // Dictionary (col 5): key width 3.
    bad = good;
    bad[desc(5) + 50] = 3;
    EXPECT_FALSE(bolt_wire_deserialize(bad.data(), bad.size(), &dst, &a));
    // List (col 6): decreasing offsets.
    bad = good;
    std::memcpy(&o2, bad.data() + desc(6) + 32, 8);
    int32_t big = 1 << 20;
    std::memcpy(bad.data() + o2 + 8, &big, 4);
    EXPECT_FALSE(bolt_wire_deserialize(bad.data(), bad.size(), &dst, &a));
    // Struct (col 7): child count mismatch.
    bad = good;
    bad[desc(7) + 52] = 4;
    EXPECT_FALSE(bolt_wire_deserialize(bad.data(), bad.size(), &dst, &a));
    // Unchanged bytes still parse.
    EXPECT_TRUE(bolt_wire_deserialize(good.data(), good.size(), &dst, &a));
}

TEST(WireKinds, NestingDepthIsBounded) {
    Arena a;
    const int64_t n = 2;
    BoltColumn cur = BoltColumn::make_flat_alloc(n, BoltType::Int64, &a);
    std::memset(cur.data, 0, 16);
    for (uint32_t d = 0; d < kWireMaxNestDepth + 2; ++d) cur = BoltColumn::make_struct(&cur, 1, n, nullptr, &a);
    BoltBatch b;
    BoltBatch::init_empty(&b);
    ASSERT_TRUE(BoltBatch::alloc_columns(&b, &a, 1));
    b.num_rows = n;
    b.schema.num_fields = 1;
    std::memset(&b.schema.fields[0], 0, sizeof(BoltField));
    b.schema.fields[0].type = BoltType::Struct;
    b.columns[0][0] = b.columns[1][0] = cur;
    EXPECT_EQ(bolt_wire_size(&b), 0u);    // refused, never unbounded recursion
}

TEST(WireKinds, KindsRideInFrames) {
    Kit k;
    build_all_kinds(&k);
    BoltBatch src;
    k.batch(&src);
    Buf file(1 << 20);
    std::vector<FrameIndexEntry> idx(4);
    std::vector<ZoneMap> roll(64);
    std::vector<uint8_t> kinds(64);
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kSpill, 0, 0, 1, idx.data(), 4,
                               roll.data(), kinds.data(), 64), FrameStatus::kOk);
    FrameMeta m;
    std::memset(&m, 0, sizeof(m));
    m.kind = FrameOpKind::kBatch;
    ASSERT_EQ(frame_file_append(&w, &src, m, FrameZones::kCompute, nullptr, nullptr), FrameStatus::kOk);
    const size_t len = frame_file_seal(&w);
    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    FrameView fv;
    ASSERT_EQ(frame_file_frame(v, 0, true, &fv), FrameStatus::kOk);
    Arena a;
    BoltBatch dst;
    ASSERT_TRUE(bolt_wire_view(fv.payload, fv.payload_len, &dst, &a));
    expect_same(src, dst);
}

// ---------------------------------------------------------------------------
// Review additions (2026-10-02): a 20-seed sweep of every new kind / type
// over 0..65,536 rows with and without validity, Nested three levels deep
// with nulls at every level, make_view slices, hostile blobs.
// ---------------------------------------------------------------------------

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 11; }
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }
};

uint8_t* rand_bitmap(Arena* a, Rng* r, int64_t n) {
    const size_t vb = static_cast<size_t>((n + 7) / 8) + 8;
    auto* v = static_cast<uint8_t*>(a->allocate(vb, 64));
    for (size_t i = 0; i < vb; ++i) v[i] = static_cast<uint8_t>(r->next() | r->next());
    return v;
}

// One column of kind `kind` (0..12) and n rows; `nulls` adds a validity bitmap.
BoltColumn make_kind(Kit* k, Rng* r, int kind, int64_t n, bool nulls, char* pool,
                     uint32_t* used) {
    Arena& a = k->a;
    BoltColumn c = BoltColumn::make_empty();
    switch (kind) {
        case 0: case 1: case 2: case 3: case 4: case 5: case 6: {
            const BoltType ts[7] = {BoltType::Date64, BoltType::Duration, BoltType::UUID,
                                    BoltType::IPv4, BoltType::FixedSizeBinary,
                                    BoltType::Decimal256, BoltType::Float16};
            c = BoltColumn::make_flat_alloc(n, ts[kind], &a);
            auto* d = static_cast<uint8_t*>(c.data);
            for (size_t i = 0; i < static_cast<size_t>(n) * c.type_size_bytes; ++i)
                d[i] = static_cast<uint8_t>(r->next());
            if (ts[kind] == BoltType::FixedSizeBinary) {
                c.fixed_width = static_cast<uint8_t>(1 + r->below(16));
                for (int64_t row = 0; row < n; ++row)
                    std::memset(d + row * 16 + c.fixed_width, 0, 16u - c.fixed_width);
            }
            break;
        }
        case 7: {   // Flat Binary / Symbol strings
            c = BoltColumn::make_flat_alloc(n, BoltType::Utf8, &a);
            c.type = r->below(2) ? BoltType::Binary : BoltType::Symbol;
            for (int64_t row = 0; row < n; ++row) {
                std::string v(static_cast<size_t>(r->below(24)), char('a' + r->below(26)));
                static_cast<StringView*>(c.data)[row] = k->str(v, pool, used);
            }
            c.str_overflow_base = pool;
            break;
        }
        case 8: {   // Constant: int, long string, or all-null
            const uint64_t which = r->below(3);
            if (which == 0) {
                c = BoltColumn::make_constant<int64_t>(static_cast<int64_t>(r->next()), n,
                                                       BoltType::Int64);
            } else if (which == 1) {
                StringView sv = k->str(std::string(13 + r->below(40), 'q'), pool, used);
                c = BoltColumn::make_constant(sv, n, BoltType::Utf8);
                c.str_overflow_base = pool;
            } else {
                c = BoltColumn::make_constant<int32_t>(0, n, BoltType::Int32);
                if (n > 0) {
                    c.validity = static_cast<uint8_t*>(a.allocate(static_cast<size_t>((n + 7) / 8) + 1, 64));
                    std::memset(c.validity, 0, static_cast<size_t>((n + 7) / 8) + 1);
                    c.stats.all_valid = false;
                }
            }
            return c;   // a constant's nulls are all-or-nothing here
        }
        case 9:
            return BoltColumn::make_sequence(static_cast<int64_t>(r->next() % 1000), 3, n,
                                             BoltType::Int64);
        case 10: {   // RLE int64, random run lengths
            auto* vals = a.allocate_array<int64_t>(static_cast<size_t>(n) + 1);
            auto* ends = a.allocate_array<int32_t>(static_cast<size_t>(n) + 1);
            int64_t runs = 0, at = 0;
            while (at < n) {
                at += 1 + static_cast<int64_t>(r->below(9));
                if (at > n) at = n;
                vals[runs] = static_cast<int64_t>(r->next());
                ends[runs++] = static_cast<int32_t>(at);
            }
            c = BoltColumn::make_rle(vals, runs, ends, n, BoltType::Int64, &a);
            break;
        }
        case 11: {   // Dictionary, key width 1 / 2 / 4, Int64 values
            const uint16_t kw = uint16_t(1u << r->below(3));
            const int64_t nd = 1 + static_cast<int64_t>(r->below(kw == 1 ? 200 : 3000));
            BoltColumn* dv = a.allocate_array<BoltColumn>(1);
            *dv = BoltColumn::make_flat_alloc(nd, BoltType::Int64, &a);
            for (int64_t i = 0; i < nd; ++i) static_cast<int64_t*>(dv->data)[i] = i * 7 - 3;
            c.format = ColumnFormat::Dictionary;
            c.type = BoltType::Int64;
            c.length = n;
            c.type_size_bytes = kw;
            c.data = a.allocate(static_cast<size_t>(n) * kw + 8, 64);
            for (int64_t row = 0; row < n; ++row) {
                const uint32_t key = static_cast<uint32_t>(r->below(static_cast<uint64_t>(nd)));
                std::memcpy(static_cast<uint8_t*>(c.data) + row * kw, &key, kw);
            }
            c.dict_child = dv;
            break;
        }
        default: {   // List<Int64>
            auto* offs = a.allocate_array<int32_t>(static_cast<size_t>(n) + 1);
            offs[0] = 0;
            for (int64_t row = 0; row < n; ++row) offs[row + 1] = offs[row] + int32_t(r->below(4));
            BoltColumn e = BoltColumn::make_flat_alloc(offs[n], BoltType::Int64, &a);
            for (int32_t i = 0; i < offs[n]; ++i) static_cast<int64_t*>(e.data)[i] = int64_t(r->next());
            c = BoltColumn::make_list(&e, offs, n, nullptr, &a);
            break;
        }
    }
    if (nulls && n > 0) {
        c.validity = rand_bitmap(&a, r, n);
        c.stats.all_valid = false;
    }
    return c;
}

}  // namespace

TEST(WireKinds, TwentySeedSweepToSixtyFiveThousandRows) {
    static const int64_t kLens[] = {0, 1, 2, 7, 63, 64, 65, 1023, 1024, 4095, 4096, 65535, 65536};
    const int kSeeds = 20;
    for (int seed = 0; seed < kSeeds; ++seed) {
        Rng r{0xC0FFEEull + static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull};
        const int64_t lens[2] = {kLens[seed % 13],
                                 1 + static_cast<int64_t>(r.below(65536))};
        for (int64_t n : lens) {
            for (int nulls = 0; nulls < 2; ++nulls) {
                Kit k;
                k.n = n;
                char* pool = static_cast<char*>(k.a.allocate(static_cast<size_t>(n) * 64 + 64, 64));
                uint32_t used = 0;
                for (int kind = 0; kind <= 12; ++kind) {
                    char name[16];
                    std::snprintf(name, sizeof(name), "k%d", kind);
                    k.add(name, make_kind(&k, &r, kind, n, nulls != 0, pool, &used));
                }
                BoltBatch src;
                k.batch(&src);
                const size_t need = bolt_wire_size(&src);
                ASSERT_GT(need, 0u) << "seed " << seed << " n " << n;
                Buf buf(need);
                ASSERT_EQ(bolt_wire_serialize(&src, buf.p, buf.n), need);
                Arena out;
                BoltBatch copy, view;
                ASSERT_TRUE(bolt_wire_deserialize(buf.p, need, &copy, &out)) << seed << " " << n;
                ASSERT_TRUE(bolt_wire_view(buf.p, need, &view, &out)) << seed << " " << n;
                expect_same(src, copy);
                expect_same(src, view);
                // Re-serialising either result gives the same bytes (bit exact).
                std::vector<uint8_t> again(bolt_wire_size(&view));
                ASSERT_EQ(bolt_wire_serialize(&view, again.data(), again.size()), need);
                ASSERT_EQ(std::memcmp(again.data(), buf.p, need), 0) << seed << " " << n;
            }
        }
    }
}

// List<Struct{Int32, List<Utf8>}> with a null at the outer list, the struct,
// the inner list and its elements.
TEST(WireKinds, NestedThreeLevelsWithNullsAtEveryLevel) {
    Kit k;
    Rng r{77};
    const int64_t n = k.n = 300;
    char* pool = static_cast<char*>(k.a.allocate(1 << 20, 64));
    uint32_t used = 0;
    // Level 3: Utf8 leaves.
    auto* o3 = k.a.allocate_array<int32_t>(4096);
    const int64_t n_struct = 700;
    o3[0] = 0;
    for (int64_t i = 0; i < n_struct; ++i) o3[i + 1] = o3[i] + int32_t(r.below(4));
    BoltColumn leaf = BoltColumn::make_flat_alloc(o3[n_struct], BoltType::Utf8, &k.a);
    for (int32_t i = 0; i < o3[n_struct]; ++i)
        static_cast<StringView*>(leaf.data)[i] =
            k.str(std::string(r.below(20), char('a' + i % 26)), pool, &used);
    leaf.str_overflow_base = pool;
    leaf.validity = rand_bitmap(&k.a, &r, o3[n_struct]);
    leaf.stats.all_valid = false;
    BoltColumn inner = BoltColumn::make_list(&leaf, o3, n_struct, rand_bitmap(&k.a, &r, n_struct), &k.a);
    // Level 2: Struct{Int32, inner list}.
    BoltColumn f[2];
    f[0] = BoltColumn::make_flat_alloc(n_struct, BoltType::Int32, &k.a);
    for (int64_t i = 0; i < n_struct; ++i) static_cast<int32_t*>(f[0].data)[i] = int32_t(i);
    f[0].validity = rand_bitmap(&k.a, &r, n_struct);
    f[0].stats.all_valid = false;
    f[1] = inner;
    BoltColumn st = BoltColumn::make_struct(f, 2, n_struct, rand_bitmap(&k.a, &r, n_struct), &k.a);
    // Level 1: the outer list over the structs.
    auto* o1 = k.a.allocate_array<int32_t>(static_cast<size_t>(n) + 1);
    o1[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        int32_t step = int32_t(r.below(5));
        if (o1[i] + step > n_struct) step = int32_t(n_struct - o1[i]);
        o1[i + 1] = o1[i] + step;
    }
    o1[n] = int32_t(n_struct);   // the last list takes the rest
    ASSERT_GE(o1[n], o1[n - 1]);
    k.add("deep", BoltColumn::make_list(&st, o1, n, rand_bitmap(&k.a, &r, n), &k.a));
    BoltBatch src;
    k.batch(&src);
    const size_t need = bolt_wire_size(&src);
    ASSERT_GT(need, 0u);
    Buf buf(need);
    ASSERT_EQ(bolt_wire_serialize(&src, buf.p, buf.n), need);
    Arena out;
    BoltBatch copy, view;
    ASSERT_TRUE(bolt_wire_deserialize(buf.p, need, &copy, &out));
    ASSERT_TRUE(bolt_wire_view(buf.p, need, &view, &out));
    expect_same(src, copy);
    expect_same(src, view);
    // Each level kept its own nulls.
    const BoltColumn* s2 = static_cast<const BoltColumn*>(view.col(0).data);
    const BoltColumn* l3 = static_cast<const BoltColumn*>(s2[0].data) + 1;
    const BoltColumn* lf = static_cast<const BoltColumn*>(l3->data);
    EXPECT_NE(view.col(0).validity, nullptr);
    EXPECT_NE(s2[0].validity, nullptr);
    EXPECT_NE(l3->validity, nullptr);
    EXPECT_NE(lf->validity, nullptr);
}

// make_view slices of Flat Utf8 keep their overflow base and serialise; a
// spilled column with no base is refused rather than written dangling.
TEST(WireKinds, ViewSlicesOfStringsRoundTrip) {
    Kit k;
    k.n = 40;
    char* pool = static_cast<char*>(k.a.allocate(8192, 64));
    uint32_t used = 0;
    BoltColumn s = BoltColumn::make_flat_alloc(100, BoltType::Utf8, &k.a);
    for (int64_t i = 0; i < 100; ++i)
        static_cast<StringView*>(s.data)[i] = k.str(std::string(5 + i % 20, char('A' + i % 26)), pool, &used);
    s.str_overflow_base = pool;
    s.decimal_scale = 0;
    k.add("v", BoltColumn::make_view(s, 37, 40));
    BoltBatch src;
    k.batch(&src);
    std::vector<uint8_t> buf(bolt_wire_size(&src));
    ASSERT_GT(buf.size(), 0u);
    ASSERT_EQ(bolt_wire_serialize(&src, buf.data(), buf.size()), buf.size());
    Arena out;
    BoltBatch dst;
    ASSERT_TRUE(bolt_wire_deserialize(buf.data(), buf.size(), &dst, &out));
    expect_same(src, dst);
    src.columns[0][0].str_overflow_base = nullptr;
    src.columns[1][0].str_overflow_base = nullptr;
    EXPECT_EQ(bolt_wire_size(&src), 0u);
}

// Hostile blobs: random byte flips and truncations of a blob of every kind
// never crash the parser (ASan/UBSan), and the specific overflow shapes are
// refused.
TEST(WireKinds, HostileBlobsAreRefusedNotRead) {
    Kit k;
    build_all_kinds(&k);
    BoltBatch src;
    k.batch(&src);
    std::vector<uint8_t> good(bolt_wire_size(&src));
    ASSERT_EQ(bolt_wire_serialize(&src, good.data(), good.size()), good.size());
    Buf work(good.size());
    Rng r{12345};
    const char* env = std::getenv("BOLT_WIRE_FUZZ_ITERS");
    const uint64_t iters = env ? std::strtoull(env, nullptr, 10) : 20000u;
    for (uint64_t it = 0; it < iters; ++it) {
        std::memcpy(work.p, good.data(), good.size());
        const size_t flips = 1 + r.below(4);
        for (size_t f = 0; f < flips; ++f)
            work.p[r.below(good.size())] ^= static_cast<uint8_t>(1u << r.below(8));
        const size_t len = r.below(8) == 0 ? r.below(good.size()) : good.size();
        Arena a;
        BoltBatch d;
        (void)bolt_wire_deserialize(work.p, len, &d, &a);
        Arena b;
        (void)bolt_wire_view(work.p, len, &d, &b);
    }
    // rows near 2^62 with a Constant / Sequence column: refused by the cap.
    std::vector<uint8_t> bad = good;
    const int64_t huge = int64_t(1) << 62;
    std::memcpy(bad.data() + 12, &huge, 8);
    Arena a;
    BoltBatch d;
    EXPECT_FALSE(bolt_wire_deserialize(bad.data(), bad.size(), &d, &a));
    // A Flat column whose b1 is shorter than rows x stride.
    Kit f;
    f.n = 16;
    BoltColumn i64 = BoltColumn::make_flat_alloc(16, BoltType::Int64, &f.a);
    std::memset(i64.data, 3, 128);
    f.add("i", i64);
    BoltBatch fb;
    f.batch(&fb);
    std::vector<uint8_t> fl(bolt_wire_size(&fb));
    ASSERT_EQ(bolt_wire_serialize(&fb, fl.data(), fl.size()), fl.size());
    const size_t d0 = kWireHeaderSize + kWireSchemaEntrySize;
    const uint64_t short_len = 64;
    std::memcpy(fl.data() + d0 + 24, &short_len, 8);
    EXPECT_FALSE(bolt_wire_deserialize(fl.data(), fl.size(), &d, &a));
    // A dictionary code past the dictionary, and a run end out of order.
    std::vector<uint8_t> dict = good;
    const size_t desc0 = kWireHeaderSize + src.num_cols * kWireSchemaEntrySize;
    uint64_t o1; std::memcpy(&o1, dict.data() + desc0 + 5 * kWireDescSize + 16, 8);
    const uint16_t code = 9;
    std::memcpy(dict.data() + o1, &code, 2);
    EXPECT_FALSE(bolt_wire_deserialize(dict.data(), dict.size(), &d, &a));
    std::vector<uint8_t> rle = good;
    uint64_t o2; std::memcpy(&o2, rle.data() + desc0 + 4 * kWireDescSize + 32, 8);
    const int32_t back = 60;
    std::memcpy(rle.data() + o2, &back, 4);
    EXPECT_FALSE(bolt_wire_deserialize(rle.data(), rle.size(), &d, &a));
}
