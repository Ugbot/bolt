// test_bolt_wire_kinds.cpp — B3: wire ids for the hot column kinds
// (Constant, Sequence, Dictionary, RLE, Nested List/Map/Struct), the missing
// fixed-width types, Flat Binary/Symbol, BoltLogical, version stamping, and
// zero-copy views over a 64 B-aligned buffer.

#include <gtest/gtest.h>

#include <cstdint>
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
            int64_t i = 0;
            while (ends[i] <= r) ++i;
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
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kSpill, 0, 0, idx.data(), 4,
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
