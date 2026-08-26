// bench_parquet_vs_arrow (bolt half) — head-to-head against Arrow/pyarrow.
//
// The fairness problem with "our writer vs their writer" is usually the
// DATA: two generators that agree in description and differ in bytes make
// every size comparison meaningless. This avoids it entirely -- bolt writes
// one UNCOMPRESSED parquet file, and the python half READS THAT FILE and
// re-writes it with pyarrow. Both sides then encode provably identical
// logical content, and no generator is duplicated.
//
// What is controlled:
//   * serial on both sides (bolt encode_pool = nullptr; pyarrow's writer is
//     single-threaded and its reader is pinned with use_threads=False), so
//     this compares CODECS AND ENCODINGS, not thread counts. bolt's parallel
//     encode is measured separately in bench_parquet_write.
//   * dictionary on for both, which is pyarrow's default and now bolt's
//     option.
//   * 1 MiB data pages on both.
//   * min-of-N, and the same batch object reused, so no run pays for data
//     construction.
//
// What is NOT equalised, and must be read alongside the numbers:
//   * bolt's GZIP is a FIXED-Huffman DEFLATE; Arrow's is zlib with dynamic
//     Huffman. bolt's ZSTD emits RAW literals and predefined FSE tables;
//     libzstd does Huffman literals and builds custom tables. Both are
//     deliberate scope choices documented in their headers, and both cost
//     ratio only. Expect bolt to lose on ratio there and the gap to be the
//     honest measure of what those omissions cost.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;
using Clock = std::chrono::steady_clock;

constexpr std::int64_t  kRows = 400000;
constexpr std::uint32_t kCols = 16;
constexpr int kRepeats = 5;

struct Rng {
    std::uint64_t s;
    std::uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<std::uint32_t>(s >> 32);
    }
};

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bolt::BoltType type_of(std::uint32_t c) {
    switch (c % 4u) {
        case 0: return bolt::BoltType::Int64;      // sorted: delta's best case
        case 1: return bolt::BoltType::Utf8;       // low cardinality
        case 2: return bolt::BoltType::Float64;
        default: return bolt::BoltType::Int64;     // random
    }
}

std::string str_val(std::uint32_t c, std::int64_t i) {
    char b[64];
    if ((c % 8u) == 1u) {
        std::snprintf(b, sizeof(b), "cat-%lld", static_cast<long long>(i % 64));
    } else {
        std::snprintf(b, sizeof(b), "id-%08lld-%u", static_cast<long long>(i), c);
    }
    return b;
}

void build_batch(bolt::Arena* a, bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = kCols;
    out->num_rows = kRows;
    bolt::BoltBatch::alloc_columns(out, a, kCols);
    Rng r{0xC0FFEE};
    char name[32];
    for (std::uint32_t c = 0; c < kCols; ++c) {
        std::snprintf(name, sizeof(name), "c%02u", c);
        const bolt::BoltType t = type_of(c);
        out->schema.add_field(name, t, false);
        bolt::BoltColumn& col = out->columns[out->read_epoch][c];
        if (t == bolt::BoltType::Utf8) {
            col = bolt::BoltColumn::make_empty();
            col.length = kRows;
            col.format = bolt::ColumnFormat::Flat;
            col.type = t;
            col.type_size_bytes = sizeof(bolt::StringView);
            auto* svs = static_cast<bolt::StringView*>(a->allocate(
                static_cast<std::size_t>(kRows) * sizeof(bolt::StringView),
                alignof(bolt::StringView)));
            std::memset(svs, 0,
                        static_cast<std::size_t>(kRows) * sizeof(bolt::StringView));
            std::size_t need = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
                if (s.size() > 12u) need += s.size();
            }
            auto* spill = need ? static_cast<std::uint8_t*>(a->allocate(need, 8))
                               : nullptr;
            std::size_t off = 0;
            for (std::int64_t i = 0; i < kRows; ++i) {
                const std::string s = str_val(c, i);
                svs[i].length = static_cast<std::uint32_t>(s.size());
                if (s.size() <= 12u) {
                    std::memcpy(&svs[i].prefix[0], s.data(), s.size());
                } else {
                    std::memcpy(&svs[i].prefix[0], s.data(), 4);
                    std::memcpy(spill + off, s.data(), s.size());
                    svs[i].ref.offset = static_cast<std::uint32_t>(off);
                    off += s.size();
                }
            }
            col.data = svs;
            col.str_overflow_base = spill;
            col.stats.all_valid = true;
        } else {
            col = bolt::BoltColumn::make_flat_alloc(kRows, t, a);
            if (t == bolt::BoltType::Float64) {
                auto* p = static_cast<double*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = static_cast<double>(i) * 0.5 + (r.next() % 100);
                }
            } else {
                auto* p = static_cast<std::int64_t*>(col.data);
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = ((c % 4u) == 0u) ? (i * 3 + c)
                                            : static_cast<std::int64_t>(r.next());
                }
            }
        }
    }
}

ParquetWriteOpts opts_for(std::uint8_t codec) {
    ParquetWriteOpts o{};
    o.n_columns = kCols;
    o.compression = codec;
    o.emit_statistics = true;
    o.use_dictionary = true;          // pyarrow's default; match it
    o.encode_pool = nullptr;          // serial: this compares codecs
    char name[32];
    for (std::uint32_t i = 0; i < kCols; ++i) {
        std::snprintf(name, sizeof(name), "c%02u", i);
        std::strncpy(o.columns[i].name, name, sizeof(o.columns[i].name) - 1);
        o.columns[i].type = type_of(i);
        o.columns[i].nullable = false;
    }
    return o;
}

bool write_file(const char* path, std::uint8_t codec, bolt::BoltBatch* b,
                double* enc_ms, std::uint64_t* bytes) {
    double best = 1e30;
    for (int rep = 0; rep < kRepeats; ++rep) {
        const auto o = opts_for(codec);
        const auto t0 = Clock::now();
        ParquetWriter* w = parquet_write_open(path, &o);
        if (w == nullptr) return false;
        if (!parquet_write_row_group(w, b)) return false;
        if (!parquet_write_close(w)) return false;
        const double ms = ms_since(t0);
        if (ms < best) best = ms;
    }
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    *bytes = static_cast<std::uint64_t>(std::ftell(f));
    std::fclose(f);
    *enc_ms = best;
    return true;
}

std::vector<std::uint8_t> slurp(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

// Decode a file with bolt, min-of-N. -1 when bolt cannot read it at all.
double read_ms(const char* path) {
    const auto buf = slurp(path);
    if (buf.empty()) return -1.0;
    double best = 1e30;
    for (int rep = 0; rep < kRepeats; ++rep) {
        bolt::Arena ra;
        auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
        const auto t0 = Clock::now();
        if (!parquet_read_file(buf.data(), buf.size(), &ra, rb)) return -1.0;
        const double ms = ms_since(t0);
        if (ms < best) best = ms;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : ".";
    bolt::Arena arena;
    auto* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);

    struct C { const char* name; std::uint8_t id; };
    const C codecs[] = {{"none", 0}, {"snappy", 1}, {"gzip", 2},
                        {"zstd", 3}, {"lz4", 4}};
    // Machine-readable so the python half can merge without re-parsing a
    // human table: one PSV line per codec.
    std::printf("#bolt|codec|enc_ms|bytes|dec_ms\n");
    for (const C& c : codecs) {
        const std::string path = dir + "/vsarrow_bolt_" + c.name + ".parquet";
        double enc = 0.0;
        std::uint64_t bytes = 0;
        if (!write_file(path.c_str(), c.id, batch, &enc, &bytes)) {
            std::printf("bolt|%s|-1|0|-1\n", c.name);
            continue;
        }
        const double dec = read_ms(path.c_str());
        std::printf("bolt|%s|%.1f|%llu|%.1f\n", c.name, enc,
                    static_cast<unsigned long long>(bytes), dec);
    }
    // Also time bolt READING each pyarrow-written file, if the python half
    // has already produced them. That is the cross-decode half of the matrix:
    // it separates "is bolt's writer producing good bytes" from "is bolt's
    // reader fast".
    for (const C& c : codecs) {
        const std::string path = dir + "/vsarrow_pa_" + c.name + ".parquet";
        const double dec = read_ms(path.c_str());
        if (dec >= 0.0) std::printf("bolt_reads_pa|%s|0|0|%.1f\n", c.name, dec);
    }
    return 0;
}
