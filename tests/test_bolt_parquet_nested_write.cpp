// Nested STRUCT / LIST / MAP writing (ParquetWriteColumn::nested).
//
// Each fixture is a schema tree plus rows of a small value model. The model
// builds the BoltColumns, and also writes "<fixture>.expect.json" holding the
// expected rows, field ids and per-leaf paths/levels. The value verdict is
// scripts/parquet_nested_write_interop.py: pyarrow and DuckDB read every
// fixture independently and must reproduce the JSON exactly. This file checks
// what bolt's own reader can see (leaf paths, levels, field ids, page index,
// struct-only values) and the refusals.

#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_pageindex.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;
using bolt::BoltColumn;
using bolt::BoltType;

// ---- value model -----------------------------------------------------------

struct V {
    enum class K : std::uint8_t { Null, Int, Real, Bool, Str, Seq } k = K::Null;
    std::int64_t   i = 0;
    double         f = 0.0;
    bool           b = false;
    std::string    s;
    std::vector<V> kids;   // list elements | struct fields | map k0,v0,k1,v1..
};

V N() { return V{}; }
V I(std::int64_t x) { V v; v.k = V::K::Int; v.i = x; return v; }
V F(double x) { V v; v.k = V::K::Real; v.f = x; return v; }
V B(bool x) { V v; v.k = V::K::Bool; v.b = x; return v; }
V S(std::string x) { V v; v.k = V::K::Str; v.s = std::move(x); return v; }
V Q(std::vector<V> kids) { V v; v.k = V::K::Seq; v.kids = std::move(kids); return v; }

const V kNullV{};

// Schema spec; flattened breadth-first so children are contiguous and after
// their parent.
struct NS {
    std::string     name;
    BoltType        t;
    bool            nullable;
    std::int32_t    fid;          // < 0: no field id
    std::vector<NS> kids;
    std::uint8_t    precision = 0, scale = 0, integer_bits = 0;
    bool            integer_signed = false, ts_local = false;
    std::uint32_t   fixed_len = 0;
};

NS leaf(std::string n, BoltType t, bool nullable, std::int32_t fid) {
    return NS{std::move(n), t, nullable, fid, {}};
}
NS grp(std::string n, BoltType t, bool nullable, std::int32_t fid, std::vector<NS> k) {
    return NS{std::move(n), t, nullable, fid, std::move(k)};
}

bool is_group(BoltType t) {
    return t == BoltType::Struct || t == BoltType::List || t == BoltType::Map;
}

std::vector<ParquetNestedNode> flatten(const NS& root) {
    std::vector<const NS*> order{&root};
    std::vector<ParquetNestedNode> out;
    for (std::size_t i = 0; i < order.size(); ++i) {
        const NS& n = *order[i];
        ParquetNestedNode nd{};
        std::snprintf(nd.name, sizeof(nd.name), "%s", n.name.c_str());
        nd.type = n.t;
        nd.nullable = n.nullable;
        nd.has_field_id = n.fid >= 0;
        nd.field_id = n.fid;
        nd.precision = n.precision;
        nd.scale = n.scale;
        nd.integer_bits = n.integer_bits;
        nd.integer_signed = n.integer_signed;
        nd.ts_local = n.ts_local;
        nd.fixed_len = n.fixed_len;
        nd.n_children = static_cast<std::uint16_t>(n.kids.size());
        nd.first_child = static_cast<std::uint16_t>(order.size());
        for (const NS& k : n.kids) order.push_back(&k);
        out.push_back(nd);
    }
    return out;
}

// ---- BoltColumn construction ----------------------------------------------

std::uint8_t* bitmap(bolt::Arena* a, const std::vector<const V*>& vals) {
    auto* bm = static_cast<std::uint8_t*>(a->allocate_zeroed(vals.size() / 8 + 1, 1));
    for (std::size_t r = 0; r < vals.size(); ++r) {
        if (vals[r]->k != V::K::Null) bm[r >> 3] |= static_cast<std::uint8_t>(1u << (r & 7));
    }
    return bm;
}

BoltColumn build_bytes(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a) {
    std::size_t total = 1;
    for (const V* v : vals) total += v->s.size();
    auto* pool = static_cast<std::uint8_t*>(a->allocate_zeroed(total, 1));
    auto* offs = a->allocate_array<std::int32_t>(vals.size() + 1);
    std::int32_t at = 0;
    offs[0] = 0;
    for (std::size_t r = 0; r < vals.size(); ++r) {
        std::memcpy(pool + at, vals[r]->s.data(), vals[r]->s.size());
        at += static_cast<std::int32_t>(vals[r]->s.size());
        offs[r + 1] = at;
    }
    auto* oc = a->allocate_array<BoltColumn>(1);
    *oc = BoltColumn::make_flat(offs, nullptr, static_cast<std::int64_t>(vals.size()) + 1,
                                BoltType::Int32);
    BoltColumn c = BoltColumn::make_empty();
    c.format = bolt::ColumnFormat::VarBinary;
    c.type = ns.t;
    c.data = pool;
    c.dict_child = oc;
    c.length = static_cast<std::int64_t>(vals.size());
    c.validity = bitmap(a, vals);
    return c;
}

BoltColumn build_leaf(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a) {
    if (ns.t == BoltType::Utf8 || ns.t == BoltType::Binary) return build_bytes(ns, vals, a);
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    BoltColumn c = BoltColumn::make_flat_alloc(n > 0 ? n : 1, ns.t, a);
    c.length = n;
    auto* d = static_cast<std::uint8_t*>(c.data);
    for (std::int64_t r = 0; r < n; ++r) {
        const V& v = *vals[static_cast<std::size_t>(r)];
        switch (ns.t) {
            case BoltType::Int32: case BoltType::Date32: {
                const auto x = static_cast<std::int32_t>(v.i);
                std::memcpy(d + r * 4, &x, 4); break;
            }
            case BoltType::Int64: case BoltType::Timestamp:
                std::memcpy(d + r * 8, &v.i, 8); break;
            case BoltType::Float32: {
                const auto x = static_cast<float>(v.f);
                std::memcpy(d + r * 4, &x, 4); break;
            }
            case BoltType::Float64: std::memcpy(d + r * 8, &v.f, 8); break;
            case BoltType::Bool: d[r] = v.b ? 1u : 0u; break;
            case BoltType::Decimal128: {
                std::uint8_t le[16];
                std::memcpy(le, &v.i, 8);
                std::memset(le + 8, v.i < 0 ? 0xFF : 0x00, 8);
                std::memcpy(d + r * 16, le, 16); break;
            }
            case BoltType::UUID:
                if (v.s.size() == 16) std::memcpy(d + r * 16, v.s.data(), 16);
                break;
            default: break;
        }
    }
    c.validity = bitmap(a, vals);
    return c;
}

BoltColumn build(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a);

BoltColumn build_struct(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a) {
    std::vector<BoltColumn> kids;
    for (std::size_t j = 0; j < ns.kids.size(); ++j) {
        std::vector<const V*> cv;
        for (const V* v : vals) cv.push_back(v->k == V::K::Null ? &kNullV : &v->kids[j]);
        kids.push_back(build(ns.kids[j], cv, a));
    }
    return BoltColumn::make_struct(kids.data(), static_cast<std::int64_t>(kids.size()),
                                   static_cast<std::int64_t>(vals.size()), bitmap(a, vals), a);
}

BoltColumn build_list_or_map(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a) {
    const bool map = ns.t == BoltType::Map;
    const std::size_t step = map ? 2u : 1u;
    auto* offs = a->allocate_array<std::int32_t>(vals.size() + 1);
    std::vector<const V*> k0, k1;
    offs[0] = 0;
    for (std::size_t r = 0; r < vals.size(); ++r) {
        if (vals[r]->k != V::K::Null) {
            for (std::size_t e = 0; e + step <= vals[r]->kids.size(); e += step) {
                k0.push_back(&vals[r]->kids[e]);
                if (map) k1.push_back(&vals[r]->kids[e + 1]);
            }
        }
        offs[r + 1] = static_cast<std::int32_t>(k0.size());
    }
    BoltColumn elem;
    if (map) {
        BoltColumn kv[2] = {build(ns.kids[0], k0, a), build(ns.kids[1], k1, a)};
        elem = BoltColumn::make_struct(kv, 2, static_cast<std::int64_t>(k0.size()), nullptr, a);
    } else {
        elem = build(ns.kids[0], k0, a);
    }
    BoltColumn c = BoltColumn::make_list(&elem, offs, static_cast<std::int64_t>(vals.size()),
                                         bitmap(a, vals), a);
    if (map) c.type = BoltType::Map;
    return c;
}

BoltColumn build(const NS& ns, const std::vector<const V*>& vals, bolt::Arena* a) {
    if (ns.t == BoltType::Struct) return build_struct(ns, vals, a);
    if (ns.t == BoltType::List || ns.t == BoltType::Map) return build_list_or_map(ns, vals, a);
    return build_leaf(ns, vals, a);
}

// ---- expected JSON ---------------------------------------------------------

void json_str(std::string* o, const std::string& s) {
    *o += '"';
    for (unsigned char ch : s) {
        if (ch == '"' || ch == '\\') { *o += '\\'; *o += static_cast<char>(ch); }
        else if (ch < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", ch); *o += b; }
        else *o += static_cast<char>(ch);
    }
    *o += '"';
}

std::string hex(const std::string& s) {
    std::string h;
    for (unsigned char ch : s) { char b[3]; std::snprintf(b, sizeof(b), "%02x", ch); h += b; }
    return h;
}

std::string decimal_text(std::int64_t u, std::uint8_t scale) {
    const bool neg = u < 0;
    std::string digits = std::to_string(neg ? -u : u);
    if (scale > 0) {
        while (digits.size() <= scale) digits.insert(digits.begin(), '0');
        digits.insert(digits.end() - scale, '.');
    }
    return neg ? "-" + digits : digits;
}

void to_json(std::string* o, const NS& ns, const V& v) {
    if (v.k == V::K::Null) { *o += "null"; return; }
    char b[40];
    switch (ns.t) {
        case BoltType::Struct:
            *o += '{';
            for (std::size_t j = 0; j < ns.kids.size(); ++j) {
                if (j) *o += ',';
                json_str(o, ns.kids[j].name);
                *o += ':';
                to_json(o, ns.kids[j], v.kids[j]);
            }
            *o += '}';
            return;
        case BoltType::List:
            *o += '[';
            for (std::size_t e = 0; e < v.kids.size(); ++e) {
                if (e) *o += ',';
                to_json(o, ns.kids[0], v.kids[e]);
            }
            *o += ']';
            return;
        case BoltType::Map:
            *o += '[';
            for (std::size_t e = 0; e + 1 < v.kids.size(); e += 2) {
                if (e) *o += ',';
                *o += '[';
                to_json(o, ns.kids[0], v.kids[e]);
                *o += ',';
                to_json(o, ns.kids[1], v.kids[e + 1]);
                *o += ']';
            }
            *o += ']';
            return;
        case BoltType::Float32:
            std::snprintf(b, sizeof(b), "%.17g", static_cast<double>(static_cast<float>(v.f)));
            *o += b; return;
        case BoltType::Float64:
            std::snprintf(b, sizeof(b), "%.17g", v.f); *o += b; return;
        case BoltType::Bool: *o += v.b ? "true" : "false"; return;
        case BoltType::Utf8: json_str(o, v.s); return;
        case BoltType::Binary: case BoltType::UUID: json_str(o, hex(v.s)); return;
        case BoltType::Decimal128: json_str(o, decimal_text(v.i, ns.scale)); return;
        default: *o += std::to_string(v.i); return;
    }
}

// Leaf paths (parquet dotted), levels and ids, derived independently of the
// writer from the spec; pyarrow and bolt's reader must agree with them.
struct LeafExp { std::string path; int max_def; int max_rep; std::int32_t fid; };

void leaves_of(const NS& n, const std::string& name, const std::string& prefix,
               int def, int rep, std::vector<LeafExp>* out) {
    const std::string p = prefix.empty() ? name : prefix + "." + name;
    def += n.nullable ? 1 : 0;
    if (!is_group(n.t)) { out->push_back({p, def, rep, n.fid}); return; }
    if (n.t == BoltType::Struct) {
        for (const NS& k : n.kids) leaves_of(k, k.name, p, def, rep, out);
        return;
    }
    const bool list = n.t == BoltType::List;
    const std::string rp = p + (list ? ".list" : ".key_value");
    if (list) { leaves_of(n.kids[0], "element", rp, def + 1, rep + 1, out); return; }
    leaves_of(n.kids[0], "key", rp, def + 1, rep + 1, out);
    leaves_of(n.kids[1], "value", rp, def + 1, rep + 1, out);
}

// Arrow-side field paths for every node that carries an id.
void ids_of(const NS& n, const std::string& path, std::string* o, bool* first) {
    if (n.fid >= 0) {
        if (!*first) *o += ',';
        *first = false;
        json_str(o, path);
        *o += ':' + std::to_string(n.fid);
    }
    if (n.t == BoltType::Struct) {
        for (const NS& k : n.kids) ids_of(k, path + "." + k.name, o, first);
    } else if (n.t == BoltType::List) {
        ids_of(n.kids[0], path + ".element", o, first);
    } else if (n.t == BoltType::Map) {
        ids_of(n.kids[0], path + ".key", o, first);
        ids_of(n.kids[1], path + ".value", o, first);
    }
}

// ---- fixture ---------------------------------------------------------------

struct Fixture {
    std::string           name;
    std::vector<NS>       cols;      // top-level columns, flat ones as leaves
    std::vector<std::vector<V>> rows;
    std::int64_t          expect_row_groups = 1;
    int                   sort_leaf = -1;   // expected RowGroup sorting column
};

void write_expect(const Fixture& fx, const std::string& file) {
    std::string o = "{\"file\":";
    json_str(&o, file);
    o += ",\"row_groups\":" + std::to_string(fx.expect_row_groups) +
         ",\"sort_leaf\":" + std::to_string(fx.sort_leaf) + ",\"rows\":[";
    for (std::size_t r = 0; r < fx.rows.size(); ++r) {
        if (r) o += ',';
        o += '{';
        for (std::size_t c = 0; c < fx.cols.size(); ++c) {
            if (c) o += ',';
            json_str(&o, fx.cols[c].name);
            o += ':';
            to_json(&o, fx.cols[c], fx.rows[r][c]);
        }
        o += '}';
    }
    o += "],\"field_ids\":{";
    bool first = true;
    for (const NS& c : fx.cols) ids_of(c, c.name, &o, &first);
    o += "},\"leaves\":[";
    std::vector<LeafExp> lv;
    for (const NS& c : fx.cols) leaves_of(c, c.name, "", 0, 0, &lv);
    for (std::size_t i = 0; i < lv.size(); ++i) {
        if (i) o += ',';
        o += "{\"path\":";
        json_str(&o, lv[i].path);
        o += ",\"max_def\":" + std::to_string(lv[i].max_def) +
             ",\"max_rep\":" + std::to_string(lv[i].max_rep) + '}';
    }
    o += "]}";
    std::FILE* f = std::fopen((fx.name + ".expect.json").c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fwrite(o.data(), 1, o.size(), f);
    std::fclose(f);
}

struct Built {
    bolt::Arena                                 arena;
    std::vector<std::vector<ParquetNestedNode>> tables;
    ParquetWriteOpts                            opts{};
    bolt::BoltBatch*                            batch = nullptr;
};

void set_flat(ParquetWriteColumn* pc, const NS& ns) {
    std::snprintf(pc->name, sizeof(pc->name), "%s", ns.name.c_str());
    pc->type = ns.t;
    pc->nullable = ns.nullable;
    pc->has_field_id = ns.fid >= 0;
    pc->field_id = ns.fid;
    pc->precision = ns.precision;
    pc->scale = ns.scale;
}

void build_fixture(const Fixture& fx, Built* bt) {
    bt->opts.n_columns = static_cast<std::uint32_t>(fx.cols.size());
    bt->tables.resize(fx.cols.size());
    auto* b = bt->arena.allocate_array<bolt::BoltBatch>(1);
    bolt::BoltBatch::init_empty(b);
    b->num_cols = static_cast<std::uint32_t>(fx.cols.size());
    b->num_rows = static_cast<std::int64_t>(fx.rows.size());
    bolt::BoltBatch::alloc_columns(b, &bt->arena, b->num_cols);
    for (std::size_t c = 0; c < fx.cols.size(); ++c) {
        const NS& ns = fx.cols[c];
        ParquetWriteColumn* pc = &bt->opts.columns[c];
        if (is_group(ns.t)) {
            bt->tables[c] = flatten(ns);
            pc->type = ns.t;
            pc->nested = bt->tables[c].data();
            pc->n_nested = static_cast<std::uint32_t>(bt->tables[c].size());
        } else {
            set_flat(pc, ns);
        }
        std::vector<const V*> vals;
        for (const auto& row : fx.rows) vals.push_back(&row[c]);
        b->schema.add_field(ns.name.c_str(), ns.t, ns.nullable);
        b->columns[b->read_epoch][c] = build(ns, vals, &bt->arena);
    }
    bt->batch = b;
}

std::vector<std::uint8_t> slurp(const std::string& path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    v.resize(static_cast<std::size_t>(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

// Writes the fixture (and its expect.json); checks what bolt's reader sees.
std::vector<std::uint8_t> write_fixture(const Fixture& fx,
                                        void (*tweak)(ParquetWriteOpts*) = nullptr,
                                        bolt::Scheduler* pool = nullptr) {
    Built bt;
    build_fixture(fx, &bt);
    if (tweak != nullptr) tweak(&bt.opts);
    bt.opts.encode_pool = pool;
    const std::string path = fx.name + ".parquet";
    ParquetWriter* w = parquet_write_open(path.c_str(), &bt.opts);
    EXPECT_NE(w, nullptr) << fx.name;
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, bt.batch)) << fx.name;
    EXPECT_TRUE(parquet_write_close(w)) << fx.name;
    write_expect(fx, path);
    auto buf = slurp(path);
    bolt::Arena ma;
    PqMeta meta{};
    EXPECT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta)) << fx.name;
    std::vector<LeafExp> lv;
    for (const NS& c : fx.cols) leaves_of(c, c.name, "", 0, 0, &lv);
    EXPECT_EQ(meta.n_columns, lv.size()) << fx.name;
    EXPECT_EQ(meta.n_row_groups, static_cast<std::uint32_t>(fx.expect_row_groups)) << fx.name;
    for (std::uint32_t i = 0; i < meta.n_columns && i < lv.size(); ++i) {
        EXPECT_STREQ(meta.columns[i].name, lv[i].path.c_str()) << fx.name;
        EXPECT_EQ(meta.columns[i].max_def, lv[i].max_def) << lv[i].path;
        EXPECT_EQ(meta.columns[i].max_rep, lv[i].max_rep) << lv[i].path;
        EXPECT_EQ(meta.columns[i].has_field_id != 0, lv[i].fid >= 0) << lv[i].path;
        if (lv[i].fid >= 0) EXPECT_EQ(meta.columns[i].field_id, lv[i].fid) << lv[i].path;
    }
    return buf;
}

// ---- fixtures --------------------------------------------------------------

Fixture fx_struct_flat() {
    Fixture fx{"nested_struct_flat", {}, {}};
    fx.cols.push_back(grp("s", BoltType::Struct, true, 1, {
        leaf("a", BoltType::Int32, false, 2), leaf("b", BoltType::Utf8, true, 3),
        leaf("c", BoltType::Float64, true, 4), leaf("d", BoltType::Bool, false, 5)}));
    fx.cols.push_back(leaf("id", BoltType::Int64, false, 6));
    for (int r = 0; r < 23; ++r) {
        V s = (r % 7 == 3) ? N()
            : Q({I(r * 10), (r % 4 == 1) ? N() : S("b" + std::to_string(r) + (r % 3 ? "_long_enough_to_spill" : "")),
                 (r % 5 == 2) ? N() : F(r * 0.5 - 3.0), B(r % 2 == 0)});
        fx.rows.push_back({s, I(r)});
    }
    return fx;
}

Fixture fx_list_of_struct() {
    Fixture fx{"nested_list_of_struct", {}, {}};
    fx.cols.push_back(grp("l", BoltType::List, true, 10, {
        grp("e", BoltType::Struct, true, 11, {leaf("x", BoltType::Int64, true, 12),
                                              leaf("y", BoltType::Utf8, false, 13)})}));
    for (int r = 0; r < 31; ++r) {
        if (r % 9 == 4) { fx.rows.push_back({N()}); continue; }
        std::vector<V> el;
        for (int e = 0; e < r % 4; ++e) {
            if ((r + e) % 6 == 5) { el.push_back(N()); continue; }
            el.push_back(Q({((r + e) % 5 == 1) ? N() : I(r * 100 + e), S("y" + std::to_string(e))}));
        }
        fx.rows.push_back({Q(el)});
    }
    return fx;
}

Fixture fx_struct_with_list() {
    Fixture fx{"nested_struct_with_list", {}, {}};
    fx.cols.push_back(grp("s", BoltType::Struct, false, 20, {
        grp("tags", BoltType::List, true, 21, {leaf("t", BoltType::Utf8, true, 22)}),
        leaf("n", BoltType::Int32, true, 23)}));
    for (int r = 0; r < 19; ++r) {
        V tags;
        if (r % 5 == 0) tags = N();
        else {
            std::vector<V> t;
            for (int e = 0; e < r % 3; ++e) t.push_back((r + e) % 4 == 0 ? N() : S("t" + std::to_string(r) + "." + std::to_string(e)));
            tags = Q(t);
        }
        fx.rows.push_back({Q({tags, (r % 6 == 1) ? N() : I(-r)})});
    }
    return fx;
}

Fixture fx_map_str_long() {
    Fixture fx{"nested_map_str_long", {}, {}};
    fx.cols.push_back(grp("m", BoltType::Map, true, 30, {
        leaf("k", BoltType::Utf8, false, 31), leaf("v", BoltType::Int64, true, 32)}));
    for (int r = 0; r < 27; ++r) {
        if (r % 8 == 2) { fx.rows.push_back({N()}); continue; }
        std::vector<V> kv;
        for (int e = 0; e < r % 5; ++e) {
            kv.push_back(S("key" + std::to_string(e)));
            kv.push_back((r + e) % 4 == 3 ? N() : I(r * 1000 + e));
        }
        fx.rows.push_back({Q(kv)});
    }
    return fx;
}

Fixture fx_map_int_struct() {
    Fixture fx{"nested_map_int_struct", {}, {}};
    fx.cols.push_back(grp("m", BoltType::Map, true, 40, {
        leaf("k", BoltType::Int32, false, 41),
        grp("v", BoltType::Struct, true, 42, {
            leaf("p", BoltType::Float64, true, 43),
            grp("q", BoltType::List, true, 44, {leaf("qe", BoltType::Int32, false, 45)})})}));
    for (int r = 0; r < 21; ++r) {
        if (r % 7 == 6) { fx.rows.push_back({N()}); continue; }
        std::vector<V> kv;
        for (int e = 0; e < r % 3; ++e) {
            kv.push_back(I(r * 10 + e));
            if ((r + e) % 5 == 4) { kv.push_back(N()); continue; }
            std::vector<V> q;
            for (int z = 0; z < (r + e) % 3; ++z) q.push_back(I(z + r));
            kv.push_back(Q({(r % 4 == 0) ? N() : F(r + e / 4.0), (r + e) % 6 == 2 ? N() : Q(q)}));
        }
        fx.rows.push_back({Q(kv)});
    }
    return fx;
}

Fixture fx_list_list() {
    Fixture fx{"nested_list_list", {}, {}};
    fx.cols.push_back(grp("ll", BoltType::List, true, 50, {
        grp("inner", BoltType::List, true, 51, {leaf("v", BoltType::Int32, true, 52)})}));
    for (int r = 0; r < 25; ++r) {
        if (r % 11 == 0) { fx.rows.push_back({N()}); continue; }
        std::vector<V> outer;
        for (int j = 0; j < r % 4; ++j) {
            if ((r + j) % 7 == 0) { outer.push_back(N()); continue; }
            std::vector<V> in;
            for (int k = 0; k < (j + r) % 3; ++k) in.push_back((r + j + k) % 5 == 0 ? N() : I(r * 100 + j * 10 + k));
            outer.push_back(Q(in));
        }
        fx.rows.push_back({Q(outer)});
    }
    return fx;
}

Fixture fx_struct_struct() {
    Fixture fx{"nested_struct_struct", {}, {}};
    fx.cols.push_back(leaf("pre", BoltType::Utf8, true, 59));
    fx.cols.push_back(grp("s", BoltType::Struct, true, 60, {
        grp("inner", BoltType::Struct, true, 61, {
            grp("deep", BoltType::Struct, true, 62, {leaf("v", BoltType::Int64, true, 63)}),
            leaf("w", BoltType::Utf8, true, 64)}),
        leaf("z", BoltType::Int32, false, 65)}));
    for (int r = 0; r < 17; ++r) {
        V deep = (r % 4 == 1) ? N() : Q({(r % 3 == 0) ? N() : I(r * r)});
        V inner = (r % 5 == 2) ? N() : Q({deep, (r % 2) ? S("w" + std::to_string(r)) : N()});
        fx.rows.push_back({(r % 6 == 0) ? N() : S("p" + std::to_string(r)),
                           (r % 8 == 7) ? N() : Q({inner, I(r)})});
    }
    return fx;
}

NS with(NS n, void (*f)(NS*)) { f(&n); return n; }

Fixture fx_all_types() {
    Fixture fx{"nested_all_types", {}, {}};
    std::vector<NS> f;
    f.push_back(with(leaf("i8", BoltType::Int32, true, 101), [](NS* n) { n->integer_bits = 8; n->integer_signed = true; }));
    f.push_back(with(leaf("i16", BoltType::Int32, true, 102), [](NS* n) { n->integer_bits = 16; n->integer_signed = true; }));
    f.push_back(leaf("i32", BoltType::Int32, true, 103));
    f.push_back(leaf("i64", BoltType::Int64, true, 104));
    f.push_back(leaf("f32", BoltType::Float32, true, 105));
    f.push_back(leaf("f64", BoltType::Float64, true, 106));
    f.push_back(leaf("bo", BoltType::Bool, true, 107));
    f.push_back(leaf("st", BoltType::Utf8, true, 108));
    f.push_back(leaf("bi", BoltType::Binary, true, 109));
    f.push_back(with(leaf("de", BoltType::Decimal128, true, 110), [](NS* n) { n->precision = 10; n->scale = 2; }));
    f.push_back(leaf("da", BoltType::Date32, true, 111));
    f.push_back(leaf("ts", BoltType::Timestamp, true, 112));
    f.push_back(with(leaf("tl", BoltType::Timestamp, true, 113), [](NS* n) { n->ts_local = true; }));
    f.push_back(with(leaf("fx", BoltType::Binary, true, 114), [](NS* n) { n->fixed_len = 4; }));
    f.push_back(leaf("uu", BoltType::UUID, true, 115));
    fx.cols.push_back(grp("s", BoltType::Struct, true, 100, f));
    fx.cols.push_back(grp("dl", BoltType::List, true, 120, {
        with(leaf("e", BoltType::Decimal128, false, 121), [](NS* n) { n->precision = 18; n->scale = 4; })}));
    for (int r = 0; r < 13; ++r) {
        auto nul = [&](int k) { return (r + k) % 5 == 4; };
        std::string uu(16, '\0');
        for (int k = 0; k < 16; ++k) uu[static_cast<std::size_t>(k)] = static_cast<char>(r * 16 + k);
        std::string fxv = {static_cast<char>(r), 'a', 'b', static_cast<char>(0xF0 + r % 16)};
        std::vector<V> v = {
            nul(0) ? N() : I(r * 9 - 60), nul(1) ? N() : I(r * 2000 - 9000), nul(2) ? N() : I(r * 77777),
            nul(3) ? N() : I(r * 123456789012LL - 5), nul(4) ? N() : F(r * 1.25 - 2),
            nul(5) ? N() : F(r / 3.0), nul(6) ? N() : B(r % 3 == 0),
            nul(7) ? N() : S("str" + std::to_string(r)), nul(8) ? N() : S(std::string("\x00\x01\xFF", 3) + std::to_string(r)),
            nul(9) ? N() : I(r * 12345 - 70000), nul(10) ? N() : I(19000 + r),
            nul(11) ? N() : I(1700000000000000LL + r * 1000001LL), nul(12) ? N() : I(1600000000000000LL - r * 7LL),
            nul(13) ? N() : S(fxv), nul(14) ? N() : S(uu)};
        std::vector<V> dl;
        for (int e = 0; e < r % 3; ++e) dl.push_back(I(r * 1000003LL - e * 99999LL));
        fx.rows.push_back({(r == 6) ? N() : Q(v), (r % 4 == 3) ? N() : Q(dl)});
    }
    return fx;
}

Fixture fx_big(std::int64_t rows) {
    Fixture fx{"nested_big_multi", {}, {}};
    fx.cols.push_back(grp("l", BoltType::List, true, 2, {
        grp("e", BoltType::Struct, true, 3, {leaf("k", BoltType::Int64, false, 4),
                                             leaf("s", BoltType::Utf8, true, 5)})}));
    fx.cols.push_back(leaf("id", BoltType::Int64, false, 1));
    fx.cols.push_back(grp("m", BoltType::Map, true, 6, {
        leaf("key", BoltType::Utf8, false, 7), leaf("value", BoltType::Float64, true, 8)}));
    for (std::int64_t r = 0; r < rows; ++r) {
        std::vector<V> el, kv;
        for (std::int64_t e = 0; e < r % 6; ++e) {
            el.push_back((r + e) % 13 == 0 ? N() : Q({I(r * 10 + e), (r + e) % 3 == 0 ? N() : S("s" + std::to_string(r + e))}));
        }
        for (std::int64_t e = 0; e < r % 4; ++e) {
            kv.push_back(S("k" + std::to_string(e)));
            kv.push_back((r * e) % 7 == 3 ? N() : F(static_cast<double>(r) + e * 0.25));
        }
        fx.rows.push_back({(r % 17 == 5) ? N() : Q(el), I(r), (r % 19 == 7) ? N() : Q(kv)});
    }
    fx.expect_row_groups = (rows + 1199) / 1200;
    fx.sort_leaf = 2;   // logical column 1 is leaf 2, after l.list.e.{k,s}
    return fx;
}

void big_opts(ParquetWriteOpts* o) {
    o->row_group_max_rows = 1200;
    o->data_page_target_bytes = 4096;
    o->emit_statistics = true;
    o->emit_page_index = true;
    o->emit_size_statistics = true;
    o->compression = 3;   // ZSTD
    o->n_sorting_columns = 1;
    o->sorting_columns[0] = ParquetSortingColumn{1, false, false};
}

void v2_opts(ParquetWriteOpts* o) {
    o->data_page_v2 = true;
    o->compression = 1;   // SNAPPY
    o->emit_statistics = true;
    o->data_page_target_bytes = 4096;
}

// ---- tests -----------------------------------------------------------------

TEST(BoltParquetNestedWrite, StructOfFlatFields) {
    const Fixture fx = fx_struct_flat();
    const auto buf = write_fixture(fx);
    ASSERT_FALSE(buf.empty());
    // bolt's reader decodes max_rep == 0 leaves directly: a null struct and a
    // null field both read as a null leaf.
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    BoltColumn out[5];
    std::int64_t rows = 0;
    ASSERT_TRUE(parquet_read_row_group(buf.data(), buf.size(), &meta, 0, &ma, out, &rows));
    ASSERT_EQ(rows, 23);
    const auto* a = static_cast<const std::int32_t*>(out[0].data);
    const auto* id = static_cast<const std::int64_t*>(out[4].data);
    for (std::int64_t r = 0; r < rows; ++r) {
        const bool snull = r % 7 == 3;
        const bool valid = out[0].validity == nullptr ||
            ((out[0].validity[(out[0].validity_offset + r) >> 3] >> ((out[0].validity_offset + r) & 7)) & 1u);
        EXPECT_EQ(valid, !snull) << r;
        if (!snull) EXPECT_EQ(a[r], r * 10) << r;
        EXPECT_EQ(id[r], r);
    }
}

TEST(BoltParquetNestedWrite, ListOfStruct) { EXPECT_FALSE(write_fixture(fx_list_of_struct()).empty()); }
TEST(BoltParquetNestedWrite, StructWithList) { EXPECT_FALSE(write_fixture(fx_struct_with_list()).empty()); }
TEST(BoltParquetNestedWrite, MapStringLong) { EXPECT_FALSE(write_fixture(fx_map_str_long()).empty()); }
TEST(BoltParquetNestedWrite, MapIntStruct) { EXPECT_FALSE(write_fixture(fx_map_int_struct()).empty()); }
TEST(BoltParquetNestedWrite, ListOfList) { EXPECT_FALSE(write_fixture(fx_list_list()).empty()); }
TEST(BoltParquetNestedWrite, StructOfStruct) { EXPECT_FALSE(write_fixture(fx_struct_struct()).empty()); }
TEST(BoltParquetNestedWrite, EveryLeafType) {
    Fixture fx = fx_all_types();
    EXPECT_FALSE(write_fixture(fx, [](ParquetWriteOpts* o) { o->emit_statistics = true; }).empty());
}

TEST(BoltParquetNestedWrite, PageV2Snappy) {
    Fixture fx = fx_map_int_struct();
    fx.name = "nested_map_int_struct_v2";
    EXPECT_FALSE(write_fixture(fx, v2_opts).empty());
    Fixture fl = fx_list_list();
    fl.name = "nested_list_list_v2";
    EXPECT_FALSE(write_fixture(fl, v2_opts).empty());
}

TEST(BoltParquetNestedWrite, ManyRowGroupsAndPages) {
    const Fixture fx = fx_big(5000);
    const auto buf = write_fixture(fx, big_opts);
    ASSERT_FALSE(buf.empty());
    bolt::Arena ma;
    PqMeta meta{};
    ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
    ASSERT_EQ(meta.n_row_groups, 5u);
    std::vector<PqPageLocation> locs(4096);
    std::uint32_t multi_page_chunks = 0;
    for (std::uint32_t c = 0; c < meta.n_chunks; ++c) {
        PqOffsetIndex oi{locs.data(), static_cast<std::uint32_t>(locs.size()), 0};
        ASSERT_TRUE(pq_read_offset_index(buf.data(), buf.size(), meta.chunks[c], &oi)) << c;
        multi_page_chunks += oi.n_pages > 1u ? 1u : 0u;
    }
    EXPECT_GT(multi_page_chunks, 10u);
}

// Placement is serial and in leaf order, so a pool changes no byte.
TEST(BoltParquetNestedWrite, ParallelEncodeIsByteIdentical) {
    Fixture fx = fx_big(3000);
    fx.name = "nested_big_serial";
    const auto serial = write_fixture(fx, big_opts);
    ASSERT_FALSE(serial.empty());
    for (std::uint32_t threads : {2u, 5u}) {
        auto* s = new bolt::Scheduler();
        ASSERT_TRUE(s->init(threads));
        fx.name = "nested_big_pool" + std::to_string(threads);
        const auto par = write_fixture(fx, big_opts, s);
        s->shutdown();
        delete s;
        EXPECT_EQ(par, serial) << threads;
    }
}

TEST(BoltParquetNestedWrite, EmptyBatch) {
    Fixture fx = fx_map_int_struct();
    fx.name = "nested_empty";
    fx.rows.clear();
    EXPECT_FALSE(write_fixture(fx).empty());
}

// ---- refusals --------------------------------------------------------------

bool open_ok(const NS& col) {
    std::vector<ParquetNestedNode> t = flatten(col);
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.columns[0].type = col.t;
    o.columns[0].nested = t.data();
    o.columns[0].n_nested = static_cast<std::uint32_t>(t.size());
    ParquetWriter* w = parquet_write_open_mem(&o);
    if (w == nullptr) return false;
    bolt::Arena a;
    const std::uint8_t* out = nullptr;
    std::uint64_t n = 0;
    (void)parquet_write_close_mem(w, &a, &out, &n);
    return true;
}

TEST(BoltParquetNestedWrite, OpenRefusesMalformedTrees) {
    EXPECT_TRUE(open_ok(grp("m", BoltType::Map, true, -1, {leaf("k", BoltType::Utf8, false, -1),
                                                           leaf("v", BoltType::Int64, true, -1)})));
    // A nullable map key.
    EXPECT_FALSE(open_ok(grp("m", BoltType::Map, true, -1, {leaf("k", BoltType::Utf8, true, -1),
                                                            leaf("v", BoltType::Int64, true, -1)})));
    // Wrong arity.
    EXPECT_FALSE(open_ok(grp("l", BoltType::List, true, -1, {leaf("a", BoltType::Int64, true, -1),
                                                             leaf("b", BoltType::Int64, true, -1)})));
    EXPECT_FALSE(open_ok(grp("s", BoltType::Struct, true, -1, {})));
    // Unnamed struct field; unsupported leaf type.
    EXPECT_FALSE(open_ok(grp("s", BoltType::Struct, true, -1, {leaf("", BoltType::Int64, true, -1)})));
    EXPECT_FALSE(open_ok(grp("s", BoltType::Struct, true, -1, {leaf("a", BoltType::Int8, true, -1)})));
    // A child at or before its parent (a cycle).
    std::vector<ParquetNestedNode> t = flatten(grp("s", BoltType::Struct, true, -1,
                                                   {leaf("a", BoltType::Int64, true, -1)}));
    t[0].first_child = 0;
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.columns[0].type = BoltType::Struct;
    o.columns[0].nested = t.data();
    o.columns[0].n_nested = 2;
    EXPECT_EQ(parquet_write_open_mem(&o), nullptr);
    // Dictionary is not available for nested leaves.
    t[0].first_child = 1;
    o.columns[0].encoding = static_cast<std::uint8_t>(PqWriteEncoding::Dictionary);
    EXPECT_EQ(parquet_write_open_mem(&o), nullptr);
    // Too deep: 17 nested structs.
    NS deep = leaf("v", BoltType::Int64, true, -1);
    for (int d = 0; d < 16; ++d) deep = grp("g" + std::to_string(d), BoltType::Struct, true, -1, {deep});
    EXPECT_FALSE(open_ok(deep));
}

bool write_rows(const NS& col, const std::vector<V>& rows) {
    Fixture fx{"nested_refuse", {col}, {}};
    for (const V& v : rows) fx.rows.push_back({v});
    Built bt;
    build_fixture(fx, &bt);
    ParquetWriter* w = parquet_write_open_mem(&bt.opts);
    if (w == nullptr) return false;
    const bool ok = parquet_write_row_group(w, bt.batch);
    bolt::Arena a;
    const std::uint8_t* out = nullptr;
    std::uint64_t n = 0;
    const bool closed = parquet_write_close_mem(w, &a, &out, &n);
    return ok && closed;
}

TEST(BoltParquetNestedWrite, NullMapKeyFailsTheRowGroup) {
    const NS m = grp("m", BoltType::Map, true, -1, {leaf("k", BoltType::Utf8, false, -1),
                                                    leaf("v", BoltType::Int64, true, -1)});
    EXPECT_TRUE(write_rows(m, {Q({S("a"), I(1)}), N()}));
    EXPECT_FALSE(write_rows(m, {Q({S("a"), I(1), N(), I(2)})}));
}

TEST(BoltParquetNestedWrite, NullInRequiredNodeFailsTheRowGroup) {
    const NS s = grp("s", BoltType::Struct, true, -1, {leaf("a", BoltType::Int64, false, -1)});
    EXPECT_TRUE(write_rows(s, {Q({I(1)}), N()}));
    EXPECT_FALSE(write_rows(s, {Q({I(1)}), Q({N()})}));
    const NS rs = grp("s", BoltType::Struct, false, -1, {leaf("a", BoltType::Int64, true, -1)});
    EXPECT_FALSE(write_rows(rs, {Q({I(1)}), N()}));
    const NS l = grp("l", BoltType::List, true, -1, {leaf("e", BoltType::Int64, false, -1)});
    EXPECT_FALSE(write_rows(l, {Q({I(1), N()})}));
}

// The batch must have the declared shape.
TEST(BoltParquetNestedWrite, ShapeMismatchFailsTheRowGroup) {
    const NS s = grp("s", BoltType::Struct, true, -1, {leaf("a", BoltType::Int64, true, -1)});
    Fixture fx{"nested_shape", {s}, {{Q({I(1)})}}};
    Built bt;
    build_fixture(fx, &bt);
    bt.batch->columns[bt.batch->read_epoch][0] =
        BoltColumn::make_flat_alloc(1, BoltType::Int64, &bt.arena);
    ParquetWriter* w = parquet_write_open_mem(&bt.opts);
    ASSERT_NE(w, nullptr);
    EXPECT_FALSE(parquet_write_row_group(w, bt.batch));
    bolt::Arena a;
    const std::uint8_t* out = nullptr;
    std::uint64_t n = 0;
    EXPECT_FALSE(parquet_write_close_mem(w, &a, &out, &n));
}

}  // namespace
