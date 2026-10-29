// Nested STRUCT / LIST / MAP assembly (parquet_read_nested_column) against
// pyarrow as the oracle. scripts/make_nested_read_fixtures.py writes each
// fixture with pyarrow and dumps pyarrow's own reading of it (to_pylist) as
// text; this test renders bolt's BoltColumn tree in the same text form and
// compares every row of every top-level field, plus the field tree with its
// names and field ids.

#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_source.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/kernels/bolt_utf8.h"

namespace {

using namespace bolt;
using namespace bolt::ingest::parquet;

std::string data_path(const std::string& name) {
#ifdef BOLT_TEST_DATA_DIR
    return std::string(BOLT_TEST_DATA_DIR) + "/nested_read/" + name;
#else
    return "tests/data/nested_read/" + name;
#endif
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> v;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(n));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

std::string quote(const uint8_t* p, size_t n) {
    std::string s = "\"";
    char hex[8];
    for (size_t i = 0; i < n; ++i) {
        const uint8_t c = p[i];
        if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\') {
            s.push_back(static_cast<char>(c));
        } else {
            std::snprintf(hex, sizeof(hex), "\\x%02x", c);
            s += hex;
        }
    }
    return s + "\"";
}

std::string i128_str(__int128 v) {
    if (v == 0) return "0";
    const bool neg = v < 0;
    unsigned __int128 u = neg ? static_cast<unsigned __int128>(-(v + 1)) + 1u
                              : static_cast<unsigned __int128>(v);
    std::string s;
    while (u != 0) { s.insert(s.begin(), static_cast<char>('0' + u % 10)); u /= 10; }
    return neg ? "-" + s : s;
}

template <typename T>
T at(const BoltColumn& c, int64_t i) {
    T v;
    std::memcpy(&v, static_cast<const uint8_t*>(c.data) + static_cast<size_t>(i) * sizeof(T), sizeof(T));
    return v;
}

std::string render_leaf(const BoltColumn& c, int64_t i) {
    char b[64];
    const uint8_t* raw = static_cast<const uint8_t*>(c.data);
    switch (c.type) {
        case BoltType::Int8:  return std::to_string(at<int8_t>(c, i));
        case BoltType::Int16: return std::to_string(at<int16_t>(c, i));
        case BoltType::Int32: return std::to_string(at<int32_t>(c, i));
        case BoltType::Int64: return std::to_string(at<int64_t>(c, i));
        case BoltType::Float64:
            std::snprintf(b, sizeof(b), "%.17g", at<double>(c, i));
            return b;
        case BoltType::Date32: return "D" + std::to_string(at<int32_t>(c, i));
        case BoltType::Timestamp: return "T" + std::to_string(at<int64_t>(c, i));
        case BoltType::Duration: return "U" + std::to_string(at<int64_t>(c, i));
        case BoltType::Decimal64:
            return std::to_string(at<int64_t>(c, i)) + "e-" +
                   std::to_string(c.decimal_scale);
        case BoltType::Decimal128: {
            const uint64_t lo = at<uint64_t>(c, 2 * i);
            const int64_t hi = at<int64_t>(c, 2 * i + 1);
            const __int128 v = (static_cast<__int128>(hi) << 64) |
                               static_cast<__int128>(lo);
            return i128_str(v) + "e-" + std::to_string(c.decimal_scale);
        }
        case BoltType::UUID: return quote(raw + i * 16, 16);
        case BoltType::FixedSizeBinary: return quote(raw + i * 16, c.fixed_width);
        case BoltType::Utf8: {
            const StringView sv = at<StringView>(c, i);
            const char* p = kernels::utf8::sv_bytes(
                sv, static_cast<const char*>(c.str_overflow_base));
            return quote(reinterpret_cast<const uint8_t*>(p), sv.length);
        }
        default: return "<type " + std::to_string(static_cast<int>(c.type)) + ">";
    }
}

bool is_null(const BoltColumn& c, int64_t i) {
    if (c.validity == nullptr) return false;
    const int64_t b = c.validity_offset + i;
    return ((c.validity[b >> 3] >> (b & 7)) & 1u) == 0u;
}

std::string render(const PqSchemaTree& t, int32_t fi, const BoltColumn& c, int64_t i) {
    if (i < 0 || i >= c.length) return "<oob>";
    if (is_null(c, i)) return "null";
    const PqField& f = t.fields[fi];
    std::string s;
    if (f.kind == PqFieldKind::Struct) {
        s = "{";
        int32_t k = f.first_child;
        for (int64_t j = 0; j < c.child_count(); ++j, k = t.fields[k].next_sibling) {
            if (j) s += ",";
            s += std::string(t.fields[k].name) + ":" + render(t, k, *c.child_at(j), i);
        }
        return s + "}";
    }
    if (f.kind == PqFieldKind::List || f.kind == PqFieldKind::Map) {
        const int32_t* o = c.list_offsets();
        const BoltColumn& e = *c.list_element();
        const bool map = f.kind == PqFieldKind::Map;
        s = map ? "<" : "[";
        for (int32_t j = o[i]; j < o[i + 1]; ++j) {
            if (j != o[i]) s += ",";
            if (!map) { s += render(t, f.first_child, e, j); continue; }
            const PqField& ent = t.fields[f.first_child];
            s += render(t, ent.first_child, *e.child_at(0), j) + "=" +
                 render(t, t.fields[ent.first_child].next_sibling, *e.child_at(1), j);
        }
        return s + (map ? ">" : "]");
    }
    return render_leaf(c, i);
}

std::string render_schema(const PqSchemaTree& t, int32_t fi) {
    const PqField& f = t.fields[fi];
    std::string s = f.name;
    if (f.has_field_id) s += "#" + std::to_string(f.field_id);
    std::string kids;
    int32_t k = f.first_child;
    if (f.kind == PqFieldKind::Map) k = t.fields[k].first_child;
    const uint32_t n = f.kind == PqFieldKind::Map ? 2u : f.n_children;
    for (uint32_t j = 0; j < n; ++j, k = t.fields[k].next_sibling) {
        if (j) kids += ",";
        kids += render_schema(t, k);
    }
    switch (f.kind) {
        case PqFieldKind::Struct: return s + "{" + kids + "}";
        case PqFieldKind::List:   return s + "[" + kids + "]";
        case PqFieldKind::Map:    return s + "<" + kids + ">";
        default:                  return s;
    }
}

struct Expected {
    std::vector<std::string> schema;
    std::vector<std::vector<std::string>> rows;
};

Expected load_expected(const std::string& name) {
    Expected e;
    std::ifstream in(data_path(name + ".expected"));
    std::string line;
    while (std::getline(in, line)) {
        const size_t a = line.find('\t');
        const size_t b = line.find('\t', a + 1);
        if (a == std::string::npos || b == std::string::npos) continue;
        const size_t k = std::stoul(line.substr(a + 1, b - a - 1));
        const std::string kind = line.substr(0, a);
        const std::string text = line.substr(b + 1);
        if (kind == "schema") {
            if (e.schema.size() <= k) e.schema.resize(k + 1);
            e.schema[k] = text;
        } else {
            if (e.rows.size() <= k) e.rows.resize(k + 1);
            e.rows[k].push_back(text);
        }
    }
    return e;
}

struct PinnedMemory {
    const std::vector<uint8_t>* file;
    uint64_t calls = 0;
    uint64_t fail_call = UINT64_MAX;
    ParquetSourceStatus failure = ParquetSourceStatus::Changed;
    static ParquetSourceStatus read(void* ctx, uint64_t offset, uint32_t length,
                                    uint8_t* out) noexcept {
        auto& memory = *static_cast<PinnedMemory*>(ctx);
        ++memory.calls;
        if (memory.calls == memory.fail_call) return memory.failure;
        if (length == 0 || length > 37u || offset > memory.file->size() ||
            length > memory.file->size() - offset) return ParquetSourceStatus::IoError;
        std::memcpy(out, memory.file->data() + offset, length);
        return ParquetSourceStatus::Ok;
    }
};

// Reads every top-level field of every row group and compares with pyarrow.
// Returns the number of value lines compared.
int64_t check_file(const std::string& name, const std::string& parquet, bool ranged = false) {
    const std::vector<uint8_t> buf = slurp(parquet);
    EXPECT_FALSE(buf.empty()) << parquet;
    if (buf.empty()) return 0;
    const Expected exp = load_expected(name);
    EXPECT_FALSE(exp.schema.empty()) << name;
    Arena arena;
    PqMeta* meta = arena.allocate_array<PqMeta>(1);
    PqSchemaTree tree;
    EXPECT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    EXPECT_TRUE(parquet_read_schema_tree(buf.data(), buf.size(), &arena, &tree));
    // Budget is the largest compressed chunk or footer, strictly less than the file.
    uint32_t footer = 0;
    std::memcpy(&footer, buf.data() + buf.size() - 8u, 4);
    uint64_t capacity = footer;
    for (uint32_t i = 0; i < meta->n_chunks; ++i)
        capacity = std::max(capacity, static_cast<uint64_t>(meta->chunks[i].total_compressed_size));
    std::vector<uint8_t> scratch(static_cast<size_t>(capacity));
    PinnedMemory memory{&buf};
    ParquetByteSource source{buf.size(), &memory, PinnedMemory::read};
    ParquetSourceWorkspace workspace{scratch.data(), capacity, 37};
    if (ranged) {
        EXPECT_LT(capacity, buf.size());
        EXPECT_EQ(parquet_read_meta_source_nested(&source, &workspace, &arena, meta, &tree),
                  ParquetSourceStatus::Ok);
        std::memset(scratch.data(), 0xDA, scratch.size());
    }
    EXPECT_EQ(tree.n_top, exp.schema.size()) << name;
    if (tree.n_top != exp.schema.size()) return 0;
    int64_t compared = 0;
    for (uint32_t k = 0; k < tree.n_top; ++k) {
        EXPECT_EQ(render_schema(tree, tree.top[k]), exp.schema[k]) << name;
        std::vector<std::string> got;
        for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
            BoltColumn col;
            int64_t rows = 0;
            const bool ok = ranged
                ? parquet_read_nested_column_source(&source, &workspace, meta, &tree, g, k,
                    &arena, &col, &rows) == ParquetSourceStatus::Ok
                : parquet_read_nested_column_tree(buf.data(), buf.size(), meta, &tree, g, k,
                    &arena, &col, &rows);
            if (ranged) std::memset(scratch.data(), 0xDA, scratch.size());
            EXPECT_TRUE(ok) << name << " field " << k << " group " << g;
            if (!ok) return compared;
            for (int64_t r = 0; r < rows; ++r) got.push_back(render(tree, tree.top[k], col, r));
        }
        EXPECT_EQ(got.size(), exp.rows[k].size()) << name << " field " << k;
        for (size_t r = 0; r < got.size() && r < exp.rows[k].size(); ++r) {
            EXPECT_EQ(got[r], exp.rows[k][r]) << name << " field " << exp.schema[k] << " row " << r;
            if (got[r] != exp.rows[k][r]) break;
            ++compared;
        }
    }
    return compared;
}

class NestedOracle : public ::testing::TestWithParam<const char*> {};

TEST_P(NestedOracle, PinnedRangesMatchPyarrow) {
    const std::string name = GetParam();
    const int64_t n = check_file(name, data_path(name + ".parquet"), true);
    EXPECT_GT(n, 0);
    std::printf("[range oracle] %s: %lld values compared\n", name.c_str(), static_cast<long long>(n));
}

TEST_P(NestedOracle, MatchesPyarrow) {
    const std::string name = GetParam();
    const int64_t n = check_file(name, data_path(name + ".parquet"));
    EXPECT_GT(n, 0);
    std::printf("[oracle] %s: %lld rows compared\n", name.c_str(), static_cast<long long>(n));
}

INSTANTIATE_TEST_SUITE_P(Pyarrow, NestedOracle, ::testing::Values(
    "struct_v1_dict", "struct_v2_plain", "struct_v2_dict",
    "list_v1_dict", "list_v2_plain", "list_v2_dict", "list_item_v1",
    "map_v1_dict", "map_v2_plain", "map_v2_dict",
    // hand-built: MAP_KEY_VALUE, Spark/parquet-mr bag.array, LogicalType-only
    "legacy_map_kv", "legacy_bag_array", "legacy_logical_only"));

// The spec-constructed 2-level LIST shapes (make_legacy_list_fixtures.py).
TEST(NestedLegacy, TwoLevelShapesMatchPyarrow) {
    for (const char* name : {"legacy2_bare", "legacy2_annotated",
                             "legacy2_multifield", "legacy2_array"}) {
        const std::string p = std::string(BOLT_TEST_DATA_DIR) + "/" + name + ".parquet";
        EXPECT_GT(check_file(name, p), 0) << name;
        EXPECT_GT(check_file(name, p, true), 0) << name;
    }
}

TEST(NestedTree, FieldIdsResolveNestedColumns) {
    const std::vector<uint8_t> buf = slurp(data_path("struct_v1_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    PqSchemaTree t;
    ASSERT_TRUE(parquet_read_schema_tree(buf.data(), buf.size(), &arena, &t));
    const int32_t s = pq_field_find_id(&t, -1, 10);
    ASSERT_GE(s, 0);
    EXPECT_STREQ(t.fields[s].name, "s");
    EXPECT_EQ(t.fields[s].kind, PqFieldKind::Struct);
    EXPECT_EQ(t.fields[s].nullable, 1u);
    const int32_t c = pq_field_find_id(&t, s, 7);
    ASSERT_GE(c, 0);
    const int32_t y = pq_field_find_id(&t, c, 6);
    ASSERT_GE(y, 0);
    EXPECT_EQ(t.fields[y].kind, PqFieldKind::List);
    const int32_t e = pq_field_find_id(&t, y, 5);
    ASSERT_GE(e, 0);
    EXPECT_EQ(t.fields[e].kind, PqFieldKind::Leaf);
    EXPECT_EQ(pq_field_find_id(&t, s, 999), -1);
    // b is REQUIRED in a nullable struct: null only where the struct is.
    const int32_t b = pq_field_find_id(&t, s, 3);
    ASSERT_GE(b, 0);
    EXPECT_EQ(t.fields[b].valid_def, t.fields[s].valid_def);
    // The raw tree keeps ids on groups too, and the leaf ranges.
    const PqSchemaNode& sn = t.nodes[t.fields[s].node];
    EXPECT_EQ(sn.has_field_id, 1u);
    EXPECT_EQ(sn.field_id, 10);
    EXPECT_EQ(sn.repetition, 1u);
    EXPECT_EQ(sn.col_end - sn.col_begin, 5u);
}

TEST(NestedTree, MapEntryAndKeyValueIds) {
    const std::vector<uint8_t> buf = slurp(data_path("map_v1_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    PqSchemaTree t;
    ASSERT_TRUE(parquet_read_schema_tree(buf.data(), buf.size(), &arena, &t));
    const int32_t m = t.top[1];
    ASSERT_EQ(t.fields[m].kind, PqFieldKind::Map);
    const int32_t ent = t.fields[m].first_child;
    EXPECT_EQ(t.fields[ent].kind, PqFieldKind::Struct);
    EXPECT_EQ(t.fields[ent].nullable, 0u);
    const int32_t key = t.fields[ent].first_child;
    EXPECT_STREQ(t.fields[key].name, "key");
    EXPECT_EQ(t.fields[key].nullable, 0u);
    EXPECT_EQ(pq_field_find_id(&t, ent, t.fields[key].field_id), key);
}

// A null struct, a struct of nulls, a null list, an empty list and a list of
// a null element are five different values; spot-check the bolt shapes
// directly rather than only through the text renderer.
TEST(NestedShapes, NullVersusEmptyAreDistinct) {
    const std::vector<uint8_t> buf = slurp(data_path("list_v1_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    PqMeta* meta = arena.allocate_array<PqMeta>(1);
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    BoltColumn col;
    int64_t rows = 0;
    ASSERT_TRUE(parquet_read_nested_column(buf.data(), buf.size(), meta, 0, 1,
                                           &arena, &col, &rows));
    ASSERT_EQ(col.type, BoltType::List);
    const int32_t* o = col.list_offsets();
    // generator: row i is null when i % 9 == 0, empty when i % 9 == 4.
    EXPECT_TRUE(is_null(col, 0));
    EXPECT_FALSE(is_null(col, 4));
    EXPECT_EQ(o[4], o[5]);
    const BoltColumn& elem = *col.list_element();
    ASSERT_EQ(elem.type, BoltType::Struct);
    // row 8: element j=0 has (8+0)%8==0 -> a null STRUCT element.
    EXPECT_TRUE(is_null(elem, o[8]));
    EXPECT_GT(o[9] - o[8], 0);
}

TEST(NestedFlat, FlatFieldEqualsFlatPath) {
    const std::vector<uint8_t> buf = slurp(data_path("struct_v2_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    PqMeta* meta = arena.allocate_array<PqMeta>(1);
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    PqSchemaTree t;
    ASSERT_TRUE(parquet_read_schema_tree(buf.data(), buf.size(), &arena, &t));
    const uint32_t top = t.n_top - 1;                       // "flat"
    const PqField& f = t.fields[t.top[top]];
    ASSERT_EQ(f.kind, PqFieldKind::Leaf);
    BoltColumn a, b;
    int64_t ra = 0, rb = 0;
    ASSERT_TRUE(parquet_read_nested_column_tree(buf.data(), buf.size(), meta, &t, 1,
                                                top, &arena, &a, &ra));
    const uint16_t c = static_cast<uint16_t>(f.column);
    ASSERT_TRUE(parquet_read_row_group_cols(buf.data(), buf.size(), meta, 1, &c, 1,
                                            &arena, &b, &rb));
    ASSERT_EQ(ra, rb);
    EXPECT_EQ(a.format, ColumnFormat::Flat);
    for (int64_t i = 0; i < ra; ++i) EXPECT_EQ(render(t, t.top[top], a, i), render(t, t.top[top], b, i));
}

TEST(NestedRobust, OutOfRangeArgumentsRefuse) {
    const std::vector<uint8_t> buf = slurp(data_path("map_v2_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    PqMeta* meta = arena.allocate_array<PqMeta>(1);
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &arena, meta));
    BoltColumn col;
    int64_t rows = 0;
    EXPECT_FALSE(parquet_read_nested_column(buf.data(), buf.size(), meta, 99, 1, &arena, &col, &rows));
    EXPECT_FALSE(parquet_read_nested_column(buf.data(), buf.size(), meta, 0, 99, &arena, &col, &rows));
    EXPECT_FALSE(parquet_read_nested_column(nullptr, buf.size(), meta, 0, 1, &arena, &col, &rows));
}

// Hostile bytes: flip bytes in the data region and footer of every fixture
// shape; every call must return (true or false), never crash.
TEST(NestedRobust, CorruptBytesNeverCrash) {
    int64_t calls = 0, ok = 0;
    for (const char* name : {"struct_v1_dict", "list_v2_plain", "map_v2_dict"}) {
        std::vector<uint8_t> buf = slurp(data_path(std::string(name) + ".parquet"));
        ASSERT_FALSE(buf.empty());
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        for (int it = 0; it < 300; ++it) {
            std::vector<uint8_t> m = buf;
            for (int k = 0; k < 4; ++k) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                m[(seed >> 17) % m.size()] ^= static_cast<uint8_t>(seed >> 40) | 1u;
            }
            Arena arena;
            PqMeta* meta = arena.allocate_array<PqMeta>(1);
            PqSchemaTree t;
            if (!parquet_read_meta(m.data(), m.size(), &arena, meta)) continue;
            if (!parquet_read_schema_tree(m.data(), m.size(), &arena, &t)) continue;
            for (uint32_t k = 0; k < t.n_top; ++k) {
                for (uint32_t g = 0; g < meta->n_row_groups; ++g) {
                    BoltColumn col;
                    int64_t rows = 0;
                    calls += 1;
                    ok += parquet_read_nested_column_tree(m.data(), m.size(), meta, &t,
                                                          g, k, &arena, &col, &rows);
                }
            }
        }
    }
    EXPECT_GT(calls, 1000);
    EXPECT_GT(ok, 0);
    EXPECT_LT(ok, calls);
    std::printf("[fuzz] %lld reads, %lld succeeded\n", static_cast<long long>(calls),
                static_cast<long long>(ok));
}

TEST(NestedSource, MetadataOutputsPublishTogether) {
    const auto buf = slurp(data_path("map_v2_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena arena;
    auto* meta = arena.allocate_array<PqMeta>(1);
    *meta = PqMeta{};
    meta->n_columns = 712;
    PqSchemaTree tree{};
    tree.n_top = 713;
    std::vector<uint8_t> scratch(buf.size());
    PinnedMemory memory{&buf};
    ParquetByteSource source{buf.size(), &memory, PinnedMemory::read};
    ParquetSourceWorkspace w{scratch.data(), scratch.size(), 37};
    memory.fail_call = 3; // Exact footer read fails after both magic reads.
    EXPECT_EQ(parquet_read_meta_source_nested(&source, &w, &arena, meta, &tree),
              ParquetSourceStatus::Changed);
    EXPECT_EQ(meta->n_columns, 712u);
    EXPECT_EQ(tree.n_top, 713u);
    EXPECT_EQ(memory.calls, 3u);
    memory.fail_call = UINT64_MAX;
    const auto before = memory.calls;
    w.bytes = reinterpret_cast<uint8_t*>(&tree);
    EXPECT_EQ(parquet_read_meta_source_nested(&source, &w, &arena, meta, &tree),
              ParquetSourceStatus::BadArg);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(meta->n_columns, 712u);
    EXPECT_EQ(tree.n_top, 713u);
}

TEST(NestedSource, ErrorsNeverPublishOrRepin) {
    const auto buf = slurp(data_path("struct_v2_dict.parquet"));
    ASSERT_FALSE(buf.empty());
    Arena metadata, output;
    auto* meta = metadata.allocate_array<PqMeta>(1);
    PqSchemaTree tree{};
    std::vector<uint8_t> scratch(buf.size());
    PinnedMemory memory{&buf};
    ParquetByteSource source{buf.size(), &memory, PinnedMemory::read};
    ParquetSourceWorkspace w{scratch.data(), scratch.size(), 37};
    ASSERT_EQ(parquet_read_meta_source_nested(&source, &w, &metadata, meta, &tree),
              ParquetSourceStatus::Ok);
    BoltColumn col = BoltColumn::make_empty();
    col.length = 712;
    int64_t rows = 713;
    for (const auto failure : {ParquetSourceStatus::Changed, ParquetSourceStatus::IoError,
                              ParquetSourceStatus::Capacity}) {
        memory.failure = failure;
        // Fail after a complete earlier leaf, not merely the initial request.
        const auto& field = tree.fields[tree.top[1]];
        ASSERT_GT(field.col_end - field.col_begin, 1u);
        const auto& chunk = meta->chunks[meta->row_groups[0].chunk_off + field.col_begin];
        memory.fail_call = memory.calls +
            (static_cast<uint64_t>(chunk.total_compressed_size) + 36u) / 37u + 2u;
        EXPECT_EQ(parquet_read_nested_column_source(&source, &w, meta, &tree, 0, 1,
                   &output, &col, &rows), failure);
        EXPECT_EQ(memory.calls, memory.fail_call);
        EXPECT_EQ(col.length, 712);
        EXPECT_EQ(rows, 0);
    }
    const auto before = memory.calls;
    w.capacity = 1;
    EXPECT_EQ(parquet_read_nested_column_source(&source, &w, meta, &tree, 0, 1,
               &output, &col, &rows), ParquetSourceStatus::Capacity);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(col.length, 712);
    rows = 713;
    w.capacity = scratch.size();
    w.bytes = reinterpret_cast<uint8_t*>(tree.fields);
    EXPECT_EQ(parquet_read_nested_column_source(&source, &w, meta, &tree, 0, 1,
               &output, &col, &rows), ParquetSourceStatus::BadArg);
    EXPECT_EQ(memory.calls, before);
    EXPECT_EQ(rows, 713);
    EXPECT_EQ(col.length, 712);
}

}  // namespace
