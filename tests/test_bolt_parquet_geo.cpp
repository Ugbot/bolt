// test_bolt_parquet_geo.cpp — G2ICE-327: WKB bounds, Parquet GEOMETRY / GEOGRAPHY / VARIANT logical types and the Iceberg
// metrics of geo columns.
//
// The WKB rules are Iceberg-Java 1.12's GeometryBoundsBuilder (x / y only, NaN skipped, one malformed value drops every
// bound, closed rings of >= 4 points, members share their collection's dimension). Geography gets counts only, as
// Iceberg-Java writes (GeospatialWriter collects no bounds). The cross-check against Java and pyarrow lives in Gestalt2's
// drill (scripts/verify_g2ice327_geo.py); this file pins the encodings.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/kernels/bolt_wkb.h"
#include "bolt/lakehouse/iceberg/geo_type.h"
#include "bolt/lakehouse/iceberg/metrics.h"
#include "bolt/lakehouse/iceberg/partition_values.h"

namespace ice = bolt::lakehouse::iceberg;
namespace pq  = bolt::ingest::parquet;
namespace wkb = bolt::kernels::wkb;

namespace {

void put_u32(std::string* s, uint32_t v, bool le) {
    for (int i = 0; i < 4; ++i) s->push_back(static_cast<char>(le ? (v >> (8 * i)) : (v >> (8 * (3 - i)))));
}
void put_f64(std::string* s, double d, bool le) {
    uint8_t b[8];
    std::memcpy(b, &d, 8);
    for (int i = 0; i < 8; ++i) s->push_back(static_cast<char>(le ? b[i] : b[7 - i]));
}
std::string head(uint32_t type, bool le = true) {
    std::string s(1, le ? '\x01' : '\x00');
    put_u32(&s, type, le);
    return s;
}
std::string point(double x, double y, bool le = true) {
    std::string s = head(1, le);
    put_f64(&s, x, le);
    put_f64(&s, y, le);
    return s;
}
std::string point_z(double x, double y, double z) {
    std::string s = head(1001);
    put_f64(&s, x, true); put_f64(&s, y, true); put_f64(&s, z, true);
    return s;
}
std::string line(const std::vector<std::pair<double, double>>& p) {
    std::string s = head(2);
    put_u32(&s, static_cast<uint32_t>(p.size()), true);
    for (auto& q : p) { put_f64(&s, q.first, true); put_f64(&s, q.second, true); }
    return s;
}
std::string polygon(const std::vector<std::pair<double, double>>& ring) {
    std::string s = head(3);
    put_u32(&s, 1, true);
    put_u32(&s, static_cast<uint32_t>(ring.size()), true);
    for (auto& q : ring) { put_f64(&s, q.first, true); put_f64(&s, q.second, true); }
    return s;
}
std::string multi(uint32_t type, const std::vector<std::string>& members) {
    std::string s = head(type);
    put_u32(&s, static_cast<uint32_t>(members.size()), true);
    for (auto& m : members) s += m;
    return s;
}

bool fold(wkb::Bounds* b, const std::string& v) {
    return wkb::bounds_add(b, reinterpret_cast<const uint8_t*>(v.data()), v.size());
}

double d_at(const uint8_t* p) { double d; std::memcpy(&d, p, 8); return d; }

}  // namespace

TEST(Wkb, BoundsFollowIcebergJava) {
    wkb::Bounds b{};
    ASSERT_TRUE(fold(&b, point(1, 2)));
    ASSERT_TRUE(fold(&b, point(-3, 5, false)));                          // big-endian
    ASSERT_TRUE(fold(&b, line({{10, -1}, {0, 0}})));
    ASSERT_TRUE(fold(&b, polygon({{0, 0}, {4, 0}, {4, 4}, {0, 0}})));
    ASSERT_TRUE(fold(&b, multi(4, {point(7, 7), point(-8, 1)})));
    ASSERT_TRUE(fold(&b, multi(7, {point(0.5, 9), line({{1, 1}, {2, 2}})})));
    ASSERT_TRUE(fold(&b, point(std::nan(""), std::nan(""))));             // POINT EMPTY: skipped
    ASSERT_TRUE(fold(&b, point_z(1, 1, 100)));
    EXPECT_EQ(b.x.lo, -8); EXPECT_EQ(b.x.hi, 10);
    EXPECT_EQ(b.y.lo, -1); EXPECT_EQ(b.y.hi, 9);
    EXPECT_TRUE(b.z.have);
    EXPECT_EQ(b.z.hi, 100);
    ASSERT_EQ(b.n_types, 6u);
    const uint32_t want[6] = {1, 2, 3, 4, 7, 1001};
    for (uint32_t i = 0; i < 6; ++i) EXPECT_EQ(b.types[i], want[i]);
    uint8_t lo[16], hi[16];
    ASSERT_TRUE(wkb::iceberg_xy_bounds(b, lo, hi));
    EXPECT_EQ(d_at(lo), -8); EXPECT_EQ(d_at(lo + 8), -1);
    EXPECT_EQ(d_at(hi), 10); EXPECT_EQ(d_at(hi + 8), 9);
}

TEST(Wkb, MalformedValuesDropEveryBound) {
    const std::string bad[] = {
        polygon({{0, 0}, {4, 0}, {4, 4}}),                               // < 4 points
        polygon({{0, 0}, {4, 0}, {4, 4}, {0, 1}}),                       // not closed
        multi(4, {line({{1, 1}, {2, 2}})}),                              // a multipoint member that is not a point
        multi(7, {point_z(1, 2, 3)}),                                    // member dimension differs from the collection
        head(8) + std::string(16, '\0'),                                 // base type 8
        point(1, 2) + "x",                                               // trailing bytes
        point(1, 2).substr(0, 12),                                       // truncated
        std::string("\x02", 1) + point(1, 2).substr(1),                  // byte order 2
    };
    for (const std::string& v : bad) {
        wkb::Bounds b{};
        ASSERT_TRUE(fold(&b, point(1, 1)));
        EXPECT_FALSE(fold(&b, v));
        EXPECT_TRUE(b.incomplete);
        EXPECT_FALSE(fold(&b, point(2, 2))) << "incomplete is sticky";
        uint8_t lo[16], hi[16];
        EXPECT_FALSE(wkb::iceberg_xy_bounds(b, lo, hi));
    }
    std::string deep = point(1, 2);
    for (uint32_t i = 0; i <= wkb::k_max_depth + 1u; ++i) deep = multi(7, {deep});
    wkb::Bounds b{};
    EXPECT_FALSE(fold(&b, deep)) << "nesting past k_max_depth is refused, never recursed into";
}

TEST(Wkb, SignedZeroOrdersAsJava) {
    wkb::Bounds b{};
    ASSERT_TRUE(fold(&b, point(0.0, -0.0)));
    ASSERT_TRUE(fold(&b, point(-0.0, 0.0)));
    EXPECT_TRUE(std::signbit(b.x.lo)); EXPECT_FALSE(std::signbit(b.x.hi));
    EXPECT_TRUE(std::signbit(b.y.lo)); EXPECT_FALSE(std::signbit(b.y.hi));
}

TEST(GeoType, ParsesJavaGrammar) {
    ice::GeoTypeInfo g{};
    auto parse = [&](const char* s) { return ice::geo_type_parse(s, static_cast<uint32_t>(std::strlen(s)), &g); };
    ASSERT_TRUE(parse("geometry"));
    EXPECT_EQ(g.kind, ice::GeoKind::kGeometry);
    EXPECT_EQ(g.crs_len, 0u);
    ASSERT_TRUE(parse(" Geometry ( srid:3857 ) "));
    EXPECT_EQ(std::string(g.crs, g.crs_len), "srid:3857");
    ASSERT_TRUE(parse("geometry(ogc:crs84)"));
    EXPECT_EQ(g.crs_len, 0u) << "the default crs, any case";
    ASSERT_TRUE(parse("geography(EPSG:4326, Karney)"));
    EXPECT_EQ(g.kind, ice::GeoKind::kGeography);
    EXPECT_EQ(g.algorithm, 4u);
    EXPECT_EQ(std::string(g.crs, g.crs_len), "EPSG:4326");
    EXPECT_FALSE(parse("geometry()"));
    EXPECT_FALSE(parse("geography(a, nope)"));
    EXPECT_FALSE(parse("geometry(a, spherical)"));
    EXPECT_FALSE(parse("geometryx"));
    EXPECT_FALSE(parse("binary"));
    char out[64];
    ASSERT_TRUE(parse("geography(OGC:CRS84, vincenty)"));
    ASSERT_TRUE(ice::geo_type_name(g, out, sizeof(out)));
    EXPECT_STREQ(out, "geography(OGC:CRS84, vincenty)");
    ASSERT_TRUE(parse("geometry"));
    ASSERT_TRUE(ice::geo_type_name(g, out, sizeof(out)));
    EXPECT_STREQ(out, "geometry");
    ice::IcebergTypeInfo t{};
    ASSERT_TRUE(ice::iceberg_type_parse("geography(srid:4326)", &t));
    EXPECT_EQ(t.type, ice::IcebergType::kGeography);
    ice::IcebergTypeInfo r{};
    ice::Transform id{};
    id.kind = ice::TransformKind::kIdentity;
    EXPECT_FALSE(ice::partition_result_type(id, t, &r)) << "spec: identity is any type except geometry/geography/variant";
}

namespace {

uint8_t* bitmap(bolt::Arena* a, const std::vector<bool>& nulls) {
    const size_t n = nulls.size();
    auto* bm = static_cast<uint8_t*>(a->allocate((n + 7u) / 8u, 8u));
    std::memset(bm, 0, (n + 7u) / 8u);
    for (size_t i = 0; i < n; ++i) if (!nulls[i]) bm[i >> 3] = static_cast<uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
    return bm;
}

bolt::BoltColumn bytes_col(bolt::Arena* a, const std::vector<std::string>& v, const std::vector<bool>& nulls) {
    auto* offs = a->allocate_array<int32_t>(v.size() + 1u);
    std::string pool;
    for (size_t i = 0; i < v.size(); ++i) { offs[i] = static_cast<int32_t>(pool.size()); pool += v[i]; }
    offs[v.size()] = static_cast<int32_t>(pool.size());
    auto* data = static_cast<char*>(a->allocate(pool.size() + 1u, 8u));
    std::memcpy(data, pool.data(), pool.size());
    bolt::BoltColumn c = bolt::BoltColumn::make_var_binary(data, bitmap(a, nulls), offs, static_cast<int64_t>(v.size()),
                                                           bolt::BoltType::Binary, a);
    c.stats.all_valid = false;
    return c;
}

// Two columns, geometry (srid:3857) and geography (karney), the same four values, one null.
bool write_geo(bolt::Arena* a, const uint8_t** out, uint64_t* len, bool page_index) {
    const std::vector<std::string> v = {point(1, 2), polygon({{0, 0}, {4, 0}, {4, -4}, {0, 0}}), "", point(-5, 3)};
    const std::vector<bool> nulls = {false, false, true, false};
    pq::ParquetWriteOpts o{};
    o.n_columns = 2;
    o.emit_statistics = true;
    o.emit_page_index = page_index;
    static const char kCrs[] = "srid:3857";
    for (uint32_t i = 0; i < 2; ++i) {
        std::snprintf(o.columns[i].name, sizeof(o.columns[i].name), "%s", i == 0 ? "g" : "gg");
        o.columns[i].type = bolt::BoltType::Binary;
        o.columns[i].nullable = true;
        o.columns[i].has_field_id = true;
        o.columns[i].field_id = static_cast<int32_t>(i + 1u);
        o.columns[i].geo = static_cast<uint8_t>(i + 1u);
    }
    std::string crs(kCrs);                                               // borrowed for open only: destroyed below
    o.columns[0].geo_crs = crs.c_str();
    o.columns[0].geo_crs_len = static_cast<uint32_t>(crs.size());
    o.columns[1].geo_algorithm = 4;
    auto* b = a->allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = 2;
    b->num_rows = 4;
    bolt::BoltBatch::alloc_columns(b, a, 2);
    b->schema.add_field("g", bolt::BoltType::Binary, true);
    b->schema.add_field("gg", bolt::BoltType::Binary, true);
    b->columns[b->read_epoch][0] = bytes_col(a, v, nulls);
    b->columns[b->read_epoch][1] = bytes_col(a, v, nulls);
    pq::ParquetWriter* w = pq::parquet_write_open_mem(&o);
    crs.assign(16, 'X');
    if (w == nullptr || !pq::parquet_write_row_group(w, b)) return false;
    return pq::parquet_write_close_mem(w, a, out, len);
}

}  // namespace

TEST(ParquetGeo, LogicalTypesAndNoMinMax) {
    bolt::Arena a;
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_geo(&a, &buf, &len, true));
    if (const char* dump = std::getenv("BOLT_TEST_GEO_DUMP")) {          // for an external reader (pyarrow) to inspect
        if (FILE* f = std::fopen(dump, "wb")) { std::fwrite(buf, 1, len, f); std::fclose(f); }
    }
    auto* meta = a.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, &a, meta));
    ASSERT_EQ(meta->n_columns, 2u);
    EXPECT_EQ(meta->columns[0].logical, static_cast<int32_t>(pq::PqLogical::Geometry));
    EXPECT_EQ(meta->columns[1].logical, static_cast<int32_t>(pq::PqLogical::Geography));
    EXPECT_EQ(meta->columns[1].geo_algorithm, 4u);
    EXPECT_EQ(meta->chunks[0].null_count, 1);
    EXPECT_EQ(meta->chunks[0].min_len + meta->chunks[0].max_len, 0u) << "undefined sort order: no min / max";
    EXPECT_EQ(meta->chunks[1].min_len + meta->chunks[1].max_len, 0u);
    const std::string file(reinterpret_cast<const char*>(buf), len);
    EXPECT_NE(file.find("srid:3857"), std::string::npos) << "the crs was copied at open";
    EXPECT_EQ(file.find("XXXXXXXXXXXXXXXX"), std::string::npos);
    bolt::BoltColumn col{};
    int64_t rows = 0;
    uint16_t c0 = 0;
    ASSERT_TRUE(pq::parquet_read_row_group_cols(buf, len, meta, 0, &c0, 1u, &a, &col, &rows));
    ASSERT_EQ(rows, 4);
    const uint8_t* p = nullptr;
    int32_t n = 0;
    col.utf8_at(0, &p, &n);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(p), static_cast<size_t>(n)), point(1, 2));
}

TEST(ParquetGeo, IcebergMetricsGeometryBoxGeographyCounts) {
    bolt::Arena a;
    const uint8_t* buf = nullptr;
    uint64_t len = 0;
    ASSERT_TRUE(write_geo(&a, &buf, &len, false));
    auto* meta = a.allocate_array<pq::PqMeta>(1);
    ASSERT_TRUE(pq::parquet_read_meta(buf, len, &a, meta));
    ice::MetricsLeaf leaves[2];
    for (uint32_t i = 0; i < 2; ++i) leaves[i].field_id = static_cast<int32_t>(i + 1u);
    ice::FileStats st{};
    ASSERT_TRUE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 2, &a, &st));
    ASSERT_EQ(st.n_cols, 2u);
    const ice::ColumnStatEntry& g = st.cols[0];
    EXPECT_EQ(g.value_count, 4);
    EXPECT_EQ(g.null_count, 1);
    ASSERT_TRUE(g.has_lower && g.has_upper);
    ASSERT_EQ(g.lower_len, 16u);
    const auto* lo = reinterpret_cast<const uint8_t*>(g.lower);
    const auto* hi = reinterpret_cast<const uint8_t*>(g.upper);
    EXPECT_EQ(d_at(lo), -5); EXPECT_EQ(d_at(lo + 8), -4);
    EXPECT_EQ(d_at(hi), 4);  EXPECT_EQ(d_at(hi + 8), 3);
    EXPECT_FALSE(g.has_nan_count);
    const ice::ColumnStatEntry& gg = st.cols[1];
    EXPECT_EQ(gg.value_count, 4);
    EXPECT_FALSE(gg.has_lower) << "geography: counts only (Iceberg-Java)";
    leaves[0].mode.mode = ice::MetricsMode::kCounts;
    ice::FileStats sc{};
    ASSERT_TRUE(ice::file_metrics_from_parquet(buf, len, meta, leaves, 2, &a, &sc));
    EXPECT_FALSE(sc.cols[0].has_lower);
}

TEST(ParquetGeo, RefusesGeoOnOtherTypes) {
    pq::ParquetWriteOpts o{};
    o.n_columns = 1;
    std::snprintf(o.columns[0].name, sizeof(o.columns[0].name), "g");
    o.columns[0].type = bolt::BoltType::Utf8;
    o.columns[0].geo = 1;
    EXPECT_EQ(pq::parquet_write_open_mem(&o), nullptr);
    o.columns[0].type = bolt::BoltType::Binary;
    o.columns[0].geo = 1;
    o.columns[0].geo_algorithm = 2;                                      // an algorithm on a geometry
    EXPECT_EQ(pq::parquet_write_open_mem(&o), nullptr);
    o.columns[0].geo_algorithm = 0;
    o.columns[0].geo_crs_len = pq::kPwMaxGeoCrsBytes + 1u;
    o.columns[0].geo_crs = "x";
    EXPECT_EQ(pq::parquet_write_open_mem(&o), nullptr);
}
