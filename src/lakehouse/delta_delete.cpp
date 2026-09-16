// bolt/lakehouse/delta_delete.cpp — DELETE WHERE, copy-on-write rewrite.
//
// G2ICE-81: a file that "may contain" a matching row used to be REMOVED
// WHOLE — every non-matching row in it was silently lost. This file now
// does what the Delta spec requires for a writer with no deletion-vector
// support (WriteOptions::enable_deletion_vectors is not wired to any writer
// yet — see G2ICE-80): decode each candidate file, decide per ROW whether
// it matches the predicate, and REWRITE the file keeping only the rows that
// don't. A file is removed whole only when every one of its rows actually
// matches; a file that turns out (despite the stats-based "may contain"
// prefilter) to have zero real matches is left untouched.
//
// Tiger Style: PODs, ≥2 asserts/fn, ≤70-line fns, no exceptions, Arena
// allocs, one commit at the very end so the operation is all-or-nothing —
// any row/column this file can't safely evaluate aborts the WHOLE delete
// (returns false) before anything is written, rather than guess.

#include "bolt/lakehouse/delta/writer.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_branchless.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/kernels/bolt_utf8.h"
#include "bolt/lakehouse/delta/snapshot.h"
#include "bolt/lakehouse/handle.h"
#include "delta_write_internal.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace pq = bolt::ingest::parquet;
namespace ku = bolt::kernels::utf8;

namespace {

// ---------------------------------------------------------------------------
// Small POD-safe string / path helpers (local copies — see delta_writer.cpp
// for the sibling versions; kept separate so this TU stays self-contained).
// ---------------------------------------------------------------------------

bool path_join(const char* a, const char* b, char* out, uint32_t cap) noexcept {
    assert(a != nullptr && b != nullptr && out != nullptr);
    const size_t al = std::strlen(a);
    const size_t bl = std::strlen(b);
    if (al + 1u + bl + 1u > cap) return false;
    std::memcpy(out, a, al);
    uint32_t p = static_cast<uint32_t>(al);
    if (al > 0 && a[al - 1] != '/' && a[al - 1] != '\\') out[p++] = '/';
    std::memcpy(out + p, b, bl);
    p += static_cast<uint32_t>(bl);
    out[p] = '\0';
    return true;
}

void str_copy_cstr(char* dst, uint32_t cap, const char* src) noexcept {
    assert(dst != nullptr && cap > 0u);
    if (src == nullptr) { dst[0] = '\0'; return; }
    const size_t n = std::strlen(src);
    const size_t m = n > cap - 1u ? cap - 1u : n;
    if (m > 0) std::memcpy(dst, src, m);
    dst[m] = '\0';
}

// A rewritten file's name MUST NOT collide with the file it is replacing:
// this call and the append path's own `make_part_uuid` are independent
// (ms-timestamp, caller-chosen small `seq`) generators, so two calls in the
// same process millisecond with the same `seq` (e.g. both "file/batch 0",
// the overwhelmingly common single-file-table shape) produce the SAME
// name. That is not merely cosmetic here: `write_rewritten_file` opens the
// new path for writing BEFORE the commit lands, so a name collision would
// physically overwrite the still-current committed file's bytes ahead of
// (and independent of) whether the commit actually succeeds -- corrupting
// the table for any reader if the commit then loses a concurrent-writer
// race. A process-wide monotonic counter makes every call from THIS
// rewriter unique regardless of clock granularity or a repeated `seq`.
std::atomic<uint64_t> g_part_uuid_nonce{0};

void make_part_uuid(uint64_t seq, char out[32]) noexcept {
    assert(out != nullptr);
    static const char hex[] = "0123456789abcdef";
    const uint64_t nonce = g_part_uuid_nonce.fetch_add(1, std::memory_order_relaxed);
    const uint64_t t = delta_writer_now_ms();
    uint64_t mix = t ^ (seq * 0x9E3779B97F4A7C15ull) ^ (nonce * 0xC2B2AE3D27D4EB4Full);
    for (uint32_t i = 0; i < 16; ++i) out[i] = hex[(mix >> ((i & 0xF) * 4)) & 0xFu];
    for (uint32_t i = 16; i < 31; ++i)
        out[i] = hex[((nonce ^ seq) >> ((i - 16) & 0xF) * 4) & 0xFu];
    out[31] = '\0';
}

const char* compression_suffix(Compression c) noexcept {
    switch (c) {
        case Compression::kSnappy: return "snappy";
        case Compression::kGzip:   return "gzip";
        case Compression::kZstd:   return "zstd";
        case Compression::kLz4:    return "lz4";
        default:                   return "raw";
    }
}

uint8_t parquet_codec_byte(Compression c) noexcept {
    return c == Compression::kSnappy ? uint8_t{1} : uint8_t{0};
}

int64_t file_size_on_disk(const char* abs_path) noexcept {
    assert(abs_path != nullptr);
    std::FILE* f = std::fopen(abs_path, "rb");
    if (f == nullptr) return -1;
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return -1; }
    const long sz = std::ftell(f);
    std::fclose(f);
    return sz < 0 ? -1 : static_cast<int64_t>(sz);
}

void ensure_parent_dir(const char* abs_path) noexcept {
    assert(abs_path != nullptr);
    char dir[kDeltaMaxPath];
    str_copy_cstr(dir, sizeof(dir), abs_path);
    char* sl = std::strrchr(dir, '/');
    if (sl == nullptr) return;
    *sl = '\0';
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
}

int32_t find_col(const pq::PqMeta* meta, const char* name) noexcept {
    assert(meta != nullptr && name != nullptr);
    for (uint32_t i = 0; i < meta->n_columns; ++i) {
        if (std::strcmp(meta->columns[i].name, name) == 0)
            return static_cast<int32_t>(i);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Row-level predicate evaluation.
// ---------------------------------------------------------------------------

bool cmp_i64(int64_t a, int64_t b, PredicateOp op) noexcept {
    switch (op) {
        case PredicateOp::kEq: return a == b;
        case PredicateOp::kNe: return a != b;
        case PredicateOp::kLt: return a < b;
        case PredicateOp::kLe: return a <= b;
        case PredicateOp::kGt: return a > b;
        case PredicateOp::kGe: return a >= b;
        default: return false;
    }
}

bool cmp_f64(double a, double b, PredicateOp op) noexcept {
    switch (op) {
        case PredicateOp::kEq: return a == b;
        case PredicateOp::kNe: return a != b;
        case PredicateOp::kLt: return a < b;
        case PredicateOp::kLe: return a <= b;
        case PredicateOp::kGt: return a > b;
        case PredicateOp::kGe: return a >= b;
        default: return false;
    }
}

bool cmp_bytes(int32_t c, PredicateOp op) noexcept {
    switch (op) {
        case PredicateOp::kEq: return c == 0;
        case PredicateOp::kNe: return c != 0;
        case PredicateOp::kLt: return c < 0;
        case PredicateOp::kLe: return c <= 0;
        case PredicateOp::kGt: return c > 0;
        case PredicateOp::kGe: return c >= 0;
        default: return false;
    }
}

// Does row `r` of `col` satisfy `pred`? `lit_i64`/`lit_f64` are the
// predicate's literal pre-resolved by value TYPE (not column type), so a
// literal typed Float64 compares correctly against an Int32 column and vice
// versa. `*supported` is set false when `col.type` has no row-level
// comparator here — the caller MUST fail the whole delete rather than treat
// an unsupported column as "matches" (data loss) or "never matches" (a
// DELETE that silently deletes nothing).
bool row_matches(const BoltColumn& col, int64_t r, const Predicate* pred,
                 int64_t lit_i64, double lit_f64, bool* supported) noexcept {
    assert(pred != nullptr && supported != nullptr);
    *supported = true;
    const bool is_null = col.is_null(r);
    if (pred->op == PredicateOp::kIsNull) return is_null;
    if (pred->op == PredicateOp::kIsNotNull) return !is_null;
    if (is_null) return false;  // NULL cmp anything -> UNKNOWN -> keep the row
    switch (col.type) {
        case BoltType::Utf8: {
            const StringView* svs = col.typed_data<StringView>();
            const StringView& sv = svs[r];
            const char* base = static_cast<const char*>(col.str_overflow_base);
            const char* bytes = ku::sv_bytes(sv, base);
            const int32_t c = ku::bytes_compare(bytes, sv.length, pred->value.str,
                                                pred->value.str_len);
            return cmp_bytes(c, pred->op);
        }
        case BoltType::Bool:
        case BoltType::Int8:
            return cmp_i64(col.typed_data<int8_t>()[r], lit_i64, pred->op);
        case BoltType::Int16:
            return cmp_i64(col.typed_data<int16_t>()[r], lit_i64, pred->op);
        case BoltType::Int32:
        case BoltType::Date32:
            return cmp_i64(col.typed_data<int32_t>()[r], lit_i64, pred->op);
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Duration:
        case BoltType::Decimal64:
            return cmp_i64(col.typed_data<int64_t>()[r], lit_i64, pred->op);
        case BoltType::UInt8:
            return cmp_i64(static_cast<int64_t>(col.typed_data<uint8_t>()[r]),
                           lit_i64, pred->op);
        case BoltType::UInt16:
            return cmp_i64(static_cast<int64_t>(col.typed_data<uint16_t>()[r]),
                           lit_i64, pred->op);
        case BoltType::UInt32:
            return cmp_i64(static_cast<int64_t>(col.typed_data<uint32_t>()[r]),
                           lit_i64, pred->op);
        case BoltType::UInt64:
            return cmp_i64(static_cast<int64_t>(col.typed_data<uint64_t>()[r]),
                           lit_i64, pred->op);
        case BoltType::Float32:
            return cmp_f64(static_cast<double>(col.typed_data<float>()[r]),
                           lit_f64, pred->op);
        case BoltType::Float64:
            return cmp_f64(col.typed_data<double>()[r], lit_f64, pred->op);
        default:
            *supported = false;
            return false;
    }
}

// ---------------------------------------------------------------------------
// Generic (type-size-dispatched) row-selection gather for one column.
// Mirrors bolt::gather_to_column<T> (bolt_column.h) but dispatches the
// element width at RUNTIME via bolt::branchless::gather_pod<N>, since a
// row group's columns span many types. Fails (false) for a column whose
// type has no fixed byte width here (Binary/List/Struct/Dictionary/...) —
// the caller must abort rather than guess how to copy it.
// ---------------------------------------------------------------------------

bool gather_pod_dispatch(const void* src, const int32_t* idx, int64_t n,
                         void* dst, uint32_t width) noexcept {
    assert(src != nullptr && dst != nullptr);
    assert(idx != nullptr || n == 0);
    switch (width) {
        case 1:  bolt::branchless::gather_pod<1>(src, idx, n, dst);  return true;
        case 2:  bolt::branchless::gather_pod<2>(src, idx, n, dst);  return true;
        case 4:  bolt::branchless::gather_pod<4>(src, idx, n, dst);  return true;
        case 8:  bolt::branchless::gather_pod<8>(src, idx, n, dst);  return true;
        case 16: bolt::branchless::gather_pod<16>(src, idx, n, dst); return true;
        case 32: bolt::branchless::gather_pod<32>(src, idx, n, dst); return true;
        default: return false;
    }
}

bool gather_column_generic(const BoltColumn& src, const int32_t* sel,
                           int64_t sel_n, Arena* arena,
                           BoltColumn* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    assert(sel != nullptr || sel_n == 0);
    if (src.format != ColumnFormat::Flat) return false;
    const size_t tsz = bolt::type_size(src.type);
    if (tsz == 0) return false;
    BoltColumn c = BoltColumn::make_flat_alloc(sel_n, src.type, arena);
    if (sel_n > 0 && c.data == nullptr) return false;
    if (sel_n > 0 &&
        !gather_pod_dispatch(src.data, sel, sel_n, c.data,
                             static_cast<uint32_t>(tsz)))
        return false;
    if (src.validity != nullptr && !src.stats.all_valid && sel_n > 0) {
        const size_t vbytes = (static_cast<size_t>(sel_n) + 7u) / 8u;
        uint8_t* nval = static_cast<uint8_t*>(arena->allocate_zeroed(vbytes));
        if (nval == nullptr) return false;
        int64_t null_count = 0;
        for (int64_t i = 0; i < sel_n; ++i) {
            const int64_t srcbit = src.validity_offset + static_cast<int64_t>(sel[i]);
            const uint8_t bit = (src.validity[srcbit >> 3] >> (srcbit & 7)) & 1u;
            nval[i >> 3] |= static_cast<uint8_t>(bit << (i & 7));
            null_count += (1 - static_cast<int64_t>(bit));
        }
        c.validity = nval;
        c.validity_offset = 0;
        c.stats.null_count = static_cast<uint32_t>(null_count);
        c.stats.all_valid = (null_count == 0);
    }
    if (src.type == BoltType::Utf8) c.str_overflow_base = src.str_overflow_base;
    c.decimal_scale = src.decimal_scale;
    *out = c;
    return true;
}

// ---------------------------------------------------------------------------
// Per-file rewrite state.
// ---------------------------------------------------------------------------

struct RgItem {
    BoltColumn* cols;    // meta.n_columns entries; nullptr => write nothing
    int64_t     n_rows;
};

// Running per-column Int64 min/max, for the new file's Delta `add.stats`.
// Matches delta_writer.cpp's build_stats_json scope (Int64 columns only) —
// not a regression, the same limitation the append path already has.
struct StatsAcc {
    bool    has_any;
    int64_t min_v;
    int64_t max_v;
};

void stats_acc_update(StatsAcc* acc, const BoltColumn& col,
                      int64_t n_rows) noexcept {
    assert(acc != nullptr);
    if (col.type != BoltType::Int64 || n_rows <= 0) return;
    const int64_t* p = col.typed_data<int64_t>();
    for (int64_t i = 0; i < n_rows; ++i) {
        if (!acc->has_any) { acc->min_v = acc->max_v = p[i]; acc->has_any = true; }
        else {
            if (p[i] < acc->min_v) acc->min_v = p[i];
            if (p[i] > acc->max_v) acc->max_v = p[i];
        }
    }
}

uint32_t build_new_stats_json(const BoltSchema* schema, const StatsAcc* accs,
                              uint32_t n_cols, int64_t num_records, char* dst,
                              uint32_t cap) noexcept {
    assert(schema != nullptr && accs != nullptr && dst != nullptr);
    assert(cap > 32u);
    int w = std::snprintf(dst, cap, "{\"numRecords\":%lld,\"minValues\":{",
                          static_cast<long long>(num_records));
    if (w < 0) return 0;
    uint32_t off = static_cast<uint32_t>(w);
    bool first = true;
    for (uint32_t c = 0; c < n_cols && off + 64u < cap; ++c) {
        if (!accs[c].has_any) continue;
        int wn = std::snprintf(dst + off, cap - off, "%s\"%s\":%lld",
                               first ? "" : ",", schema->field(static_cast<int>(c)).name,
                               static_cast<long long>(accs[c].min_v));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
        first = false;
    }
    int wn = std::snprintf(dst + off, cap - off, "},\"maxValues\":{");
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    first = true;
    for (uint32_t c = 0; c < n_cols && off + 64u < cap; ++c) {
        if (!accs[c].has_any) continue;
        wn = std::snprintf(dst + off, cap - off, "%s\"%s\":%lld",
                           first ? "" : ",", schema->field(static_cast<int>(c)).name,
                           static_cast<long long>(accs[c].max_v));
        if (wn < 0) break;
        off += static_cast<uint32_t>(wn);
        first = false;
    }
    wn = std::snprintf(dst + off, cap - off, "}}");
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return off;
}

// Decode one row group in full, evaluate the predicate column, and either
// buffer it unchanged (no row in it matched), drop it (every row matched),
// or gather the surviving rows into a fresh column set. Updates the running
// file-level matched/kept counters. Returns false on any hard failure
// (decode error, unsupported column type) — caller aborts the whole delete.
bool process_row_group(Arena* file_arena, const uint8_t* body, uint64_t blen,
                       const pq::PqMeta* meta, uint32_t rg, int32_t pred_col,
                       const Predicate* pred, int64_t lit_i64, double lit_f64,
                       RgItem* out_item, int64_t* io_matched,
                       int64_t* io_kept) noexcept {
    assert(file_arena != nullptr && body != nullptr && meta != nullptr);
    assert(out_item != nullptr && io_matched != nullptr && io_kept != nullptr);
    BoltColumn* cols = file_arena->allocate_array<BoltColumn>(meta->n_columns);
    if (cols == nullptr) return false;
    int64_t rows = 0;
    if (!pq::parquet_read_row_group(body, blen, meta, rg, file_arena, cols, &rows))
        return false;
    int32_t* keep_idx = file_arena->allocate_array<int32_t>(
        rows > 0 ? static_cast<size_t>(rows) : size_t{1});
    if (keep_idx == nullptr) return false;
    int64_t keep_n = 0, matched_n = 0;
    const BoltColumn& predcol = cols[pred_col];
    for (int64_t r = 0; r < rows; ++r) {
        bool supported = true;
        const bool m = row_matches(predcol, r, pred, lit_i64, lit_f64, &supported);
        if (!supported) return false;
        if (m) ++matched_n; else keep_idx[keep_n++] = static_cast<int32_t>(r);
    }
    *io_matched += matched_n;
    *io_kept += keep_n;
    if (matched_n == 0) { *out_item = RgItem{cols, rows}; return true; }
    if (keep_n == 0)    { *out_item = RgItem{nullptr, 0}; return true; }
    BoltColumn* gathered = file_arena->allocate_array<BoltColumn>(meta->n_columns);
    if (gathered == nullptr) return false;
    for (uint32_t c = 0; c < meta->n_columns; ++c) {
        if (!gather_column_generic(cols[c], keep_idx, keep_n, file_arena,
                                   &gathered[c]))
            return false;
    }
    *out_item = RgItem{gathered, keep_n};
    return true;
}

// Generate a `data/part-*.parquet` relative path guaranteed to name NO
// object that already exists in the table's store. `make_part_uuid`'s
// (timestamp, seq, in-process nonce) mix makes a collision astronomically
// unlikely, but "unlikely" is not the bar for a name that is about to be
// opened for writing and then referenced by an ADD action -- a collision
// with the very file this call is replacing would let the physical write
// clobber it before the commit that's supposed to gate that ever lands
// (see the collision report in the G2ICE-81 verification session: the
// first call in a process has nonce==0 and can degrade to exactly the
// append path's own (timestamp, seq) pair). So this re-checks with
// `os_head` and retries on an actual hit, bounded (never an infinite loop
// on a pathological store).
bool pick_new_rel_path(TableHandle* th, uint64_t seq, char* out_rel,
                       uint32_t rel_cap) noexcept {
    assert(th != nullptr && out_rel != nullptr && rel_cap > 0u);
    static constexpr uint32_t kMaxAttempts = 8u;
    for (uint32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
        char uuid[32];
        make_part_uuid(seq + attempt * 0x1000000ull, uuid);
        char rel[kDeltaMaxPath];
        if (std::snprintf(rel, sizeof(rel), "data/part-%s.%s.parquet", uuid,
                          compression_suffix(Compression::kSnappy)) <= 0)
            return false;
        char tbl_rel[kDeltaMaxPath];
        if (!path_join(th->table_rel, rel, tbl_rel, sizeof(tbl_rel))) return false;
        ObjectMeta om{};
        if (os_head(&th->os, tbl_rel, &om) != kOsOk || !om.exists) {
            str_copy_cstr(out_rel, rel_cap, rel);
            return true;
        }
    }
    return false;
}

bool build_write_opts(const BoltSchema* schema, uint32_t n_cols,
                      Compression compression, pq::ParquetWriteOpts* opts) noexcept {
    assert(schema != nullptr && opts != nullptr);
    assert(n_cols <= bolt::kMaxFixedColumns);
    std::memset(opts, 0, sizeof(*opts));
    opts->n_columns = n_cols;
    opts->row_group_target_bytes = 1u << 20;
    opts->compression = parquet_codec_byte(compression);
    opts->emit_statistics = false;  // Delta-log stats built separately below.
    for (uint32_t c = 0; c < n_cols; ++c) {
        const BoltField& f = schema->field(static_cast<int>(c));
        str_copy_cstr(opts->columns[c].name, sizeof(opts->columns[c].name), f.name);
        opts->columns[c].type = f.type;
        opts->columns[c].nullable = f.nullable;
    }
    return true;
}

// Write every buffered row-group item (n_rows > 0) into one new part file
// under the table's data/ prefix, accumulating Int64 stats as it goes.
// Returns false on any writer failure.
bool write_rewritten_file(TableHandle* th, const pq::PqMeta* meta,
                          const BoltSchema* wschema, const RgItem* items,
                          uint32_t n_items, uint64_t seq, char* out_rel,
                          uint32_t rel_cap, int64_t* out_size, char* out_stats,
                          uint32_t stats_cap, uint32_t* out_stats_len) noexcept {
    assert(th != nullptr && meta != nullptr && items != nullptr);
    assert(out_rel != nullptr && out_size != nullptr);
    char rel[kDeltaMaxPath];
    if (!pick_new_rel_path(th, seq, rel, sizeof(rel))) return false;
    str_copy_cstr(out_rel, rel_cap, rel);
    char tbl_rel[kDeltaMaxPath];
    if (!path_join(th->table_rel, rel, tbl_rel, sizeof(tbl_rel))) return false;
    char abs_path[kDeltaMaxPath];
    if (!path_join(th->fs_store.root, tbl_rel, abs_path, sizeof(abs_path)))
        return false;
    ensure_parent_dir(abs_path);

    pq::ParquetWriteOpts wopts{};
    if (!build_write_opts(wschema, meta->n_columns, Compression::kSnappy, &wopts))
        return false;
    pq::ParquetWriter* w = pq::parquet_write_open(abs_path, &wopts);
    if (w == nullptr) return false;

    Arena stat_arena;
    StatsAcc* accs = stat_arena.allocate_array<StatsAcc>(meta->n_columns);
    if (accs == nullptr) { pq::parquet_write_close(w); return false; }
    std::memset(accs, 0, sizeof(StatsAcc) * meta->n_columns);
    int64_t total_rows = 0;
    for (uint32_t i = 0; i < n_items; ++i) {
        if (items[i].n_rows <= 0 || items[i].cols == nullptr) continue;
        BoltBatch b{};
        BoltBatch::init_empty(&b);
        b.num_cols = meta->n_columns;
        b.num_rows = items[i].n_rows;
        b.columns[0] = items[i].cols;
        if (!pq::parquet_write_row_group(w, &b)) { pq::parquet_write_close(w); return false; }
        for (uint32_t c = 0; c < meta->n_columns; ++c)
            stats_acc_update(&accs[c], items[i].cols[c], items[i].n_rows);
        total_rows += items[i].n_rows;
    }
    if (!pq::parquet_write_close(w)) return false;
    *out_size = file_size_on_disk(abs_path);
    if (out_stats != nullptr && out_stats_len != nullptr) {
        *out_stats_len = build_new_stats_json(wschema, accs, meta->n_columns,
                                              total_rows, out_stats, stats_cap);
    }
    return *out_size >= 0;
}

uint32_t emit_remove(char* dst, uint32_t cap, uint32_t off, const char* path,
                     uint64_t ms) noexcept {
    assert(dst != nullptr && path != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"remove\":{\"path\":\"%s\",\"deletionTimestamp\":%llu,"
        "\"dataChange\":true}}\n",
        path, static_cast<unsigned long long>(ms));
    if (wn < 0) return off;
    return off + static_cast<uint32_t>(wn);
}

uint32_t emit_add(char* dst, uint32_t cap, uint32_t off, const char* path,
                  int64_t size, uint64_t ms, const char* stats_json,
                  uint32_t stats_len) noexcept {
    assert(dst != nullptr && path != nullptr);
    int wn = std::snprintf(dst + off, cap - off,
        "{\"add\":{\"path\":\"%s\",\"size\":%lld,\"modificationTime\":%llu,"
        "\"dataChange\":true,\"partitionValues\":{}",
        path, static_cast<long long>(size), static_cast<unsigned long long>(ms));
    if (wn < 0) return off;
    off += static_cast<uint32_t>(wn);
    if (stats_len > 0 && off + stats_len + 16u < cap) {
        wn = std::snprintf(dst + off, cap - off, ",\"stats\":\"");
        if (wn > 0) off += static_cast<uint32_t>(wn);
        off += delta_writer_escape_json(dst, cap, off, stats_json);
        if (off + 2u < cap) dst[off++] = '"';
    }
    if (off + 4u < cap) { dst[off++] = '}'; dst[off++] = '}'; dst[off++] = '\n'; dst[off] = '\0'; }
    return off;
}

}  // namespace

bool delta_table_delete(TableHandle* th, const Predicate* pred) noexcept {
    assert(th != nullptr && pred != nullptr);
    Arena scratch;
    Snapshot snap{};
    if (!delta_snapshot_build(&th->os, th->table_rel, -1, &scratch, &snap))
        return false;
    int64_t base = snap.version;
    const uint32_t cap = 128u * 1024u + snap.n_files * 4096u;
    char* body = scratch.allocate_array<char>(cap);
    if (body == nullptr) return false;
    uint32_t off = 0;
    const uint64_t ms = delta_writer_now_ms();
    uint32_t hits = 0;

    const bool lit_is_float = pred->value.type == BoltType::Float32 ||
                              pred->value.type == BoltType::Float64;
    const int64_t lit_i64 = lit_is_float
        ? static_cast<int64_t>(pred->value.f64) : pred->value.i64;
    const double lit_f64 = lit_is_float
        ? pred->value.f64 : static_cast<double>(pred->value.i64);

    for (uint32_t i = 0; i < snap.n_files; ++i) {
        if (!delta_file_passes(&snap.files[i], pred, 1)) continue;
        Arena file_arena;
        char key[kDeltaMaxPath];
        if (!path_join(th->table_rel, snap.files[i].path, key, sizeof(key)))
            return false;
        const uint8_t* fbody = nullptr;
        uint64_t flen = 0;
        if (os_get(&th->os, key, &file_arena, &fbody, &flen) != kOsOk) return false;
        pq::PqMeta meta{};
        if (!pq::parquet_read_meta(fbody, flen, &file_arena, &meta)) return false;
        const int32_t pred_col = find_col(&meta, pred->column);
        if (pred_col < 0) continue;  // column absent here -> cannot match

        RgItem* items = file_arena.allocate_array<RgItem>(
            meta.n_row_groups > 0 ? meta.n_row_groups : 1);
        if (items == nullptr) return false;
        int64_t total_matched = 0, total_kept = 0;
        bool ok = true;
        for (uint32_t rg = 0; rg < meta.n_row_groups && ok; ++rg) {
            ok = process_row_group(&file_arena, fbody, flen, &meta, rg,
                                   pred_col, pred, lit_i64, lit_f64, &items[rg],
                                   &total_matched, &total_kept);
        }
        if (!ok) return false;
        if (total_matched == 0) continue;  // stats false-positive; untouched

        off = emit_remove(body, cap, off, snap.files[i].path, ms);
        ++hits;
        if (total_kept == 0) continue;  // every row matched -> whole-file remove

        BoltField* fbuf = file_arena.allocate_array<BoltField>(meta.n_columns);
        if (fbuf == nullptr) return false;
        BoltSchema wschema;
        wschema.set_storage(fbuf, meta.n_columns);
        if (!pq::parquet_schema_from_meta(&meta, &wschema, false)) return false;

        char new_rel[kDeltaMaxPath];
        int64_t new_size = -1;
        char stats_json[kDeltaMaxStatsBytes];
        uint32_t stats_len = 0;
        if (!write_rewritten_file(th, &meta, &wschema, items, meta.n_row_groups,
                                  static_cast<uint64_t>(i), new_rel,
                                  sizeof(new_rel), &new_size, stats_json,
                                  sizeof(stats_json), &stats_len))
            return false;
        off = emit_add(body, cap, off, new_rel, new_size, ms, stats_json, stats_len);
    }
    if (hits == 0) return true;
    int wn = std::snprintf(body + off, cap - off,
        "{\"commitInfo\":{\"timestamp\":%llu,\"operation\":\"DELETE\"}}\n",
        static_cast<unsigned long long>(ms));
    if (wn > 0) off += static_cast<uint32_t>(wn);
    return delta_writer_commit_raw(th, body, off, &base);
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
