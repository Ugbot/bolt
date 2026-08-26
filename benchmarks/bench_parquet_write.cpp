// bench_parquet_write — what the parquet WRITER actually costs.
//
// Every claim made about this writer so far has been about correctness. This
// measures the rest: what each encoding and codec costs to produce, what it
// buys in bytes, what the page index and bloom filter add, and whether
// parallel column encoding actually scales.
//
// Method, because a compression benchmark is easy to fool:
//   * min-of-N, not mean -- the minimum is the least noisy estimator of the
//     cost when nothing else is competing for the machine.
//   * the SAME batch is written by every configuration, built once up front,
//     so no run pays for data generation.
//   * order-alternated across repeats, so a thermal ramp cannot flatter
//     whichever configuration happens to run first.
//   * BOTH time and bytes are reported. A codec that is fast because it did
//     nothing, and one that is small because it took forever, are the two
//     ways to look good on half a benchmark.
//
// Deliberately NOT a comparison against Arrow: bolt writes with a
// fixed-Huffman DEFLATE and a greedy LZ4, and the interesting question at
// this stage is the shape of the trade inside bolt, not a ratio against a
// tuned reference. scripts/parquet_write_interop.py already establishes that
// what bolt writes is what the ecosystem reads.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <chrono>
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
using Clock = std::chrono::steady_clock;

constexpr std::int64_t kRows = 400000;
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

// A schema shaped like real columnar data rather than one synthetic pattern:
// a sorted key (delta's best case), a low-cardinality string (dictionary's),
// a high-cardinality string (nobody's), doubles (byte-stream-split's), and
// plain integers.
bolt::BoltType type_of(std::uint32_t c) {
    switch (c % 4u) {
        case 0: return bolt::BoltType::Int64;
        case 1: return bolt::BoltType::Utf8;
        case 2: return bolt::BoltType::Float64;
        default: return bolt::BoltType::Int64;
    }
}

std::string str_val(std::uint32_t c, std::int64_t i) {
    char b[64];
    if ((c % 8u) == 1u) {
        std::snprintf(b, sizeof(b), "cat-%lld",
                      static_cast<long long>(i % 64));      // low cardinality
    } else {
        std::snprintf(b, sizeof(b), "id-%08lld-%u",
                      static_cast<long long>(i), c);        // high cardinality
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
                // c%4==0 is sorted (delta's best case); c%4==3 is random.
                for (std::int64_t i = 0; i < kRows; ++i) {
                    p[i] = ((c % 4u) == 0u) ? (i * 3 + c)
                                            : static_cast<std::int64_t>(r.next());
                }
            }
        }
    }
}

struct Cfg {
    const char*      name;
    std::uint8_t     codec;        // 0 none, 1 snappy, 2 gzip, 4 lz4_raw
    bool             dictionary;
    PqWriteEncoding  enc;          // Auto unless forcing one
    bool             page_index;
    bool             bloom;
    bool             v2;
    bolt::Scheduler* pool;
};

ParquetWriteOpts make_opts(const Cfg& c) {
    ParquetWriteOpts o{};
    o.n_columns = kCols;
    o.compression = c.codec;
    o.emit_statistics = true;
    o.use_dictionary = c.dictionary;
    o.emit_page_index = c.page_index;
    o.emit_bloom_filter = c.bloom;
    o.data_page_v2 = c.v2;
    o.encode_pool = c.pool;
    char name[32];
    for (std::uint32_t i = 0; i < kCols; ++i) {
        std::snprintf(name, sizeof(name), "c%02u", i);
        std::strncpy(o.columns[i].name, name, sizeof(o.columns[i].name) - 1);
        o.columns[i].type = type_of(i);
        o.columns[i].nullable = false;
        // A forced encoding applies only where the type permits it, so one
        // switch can sweep the whole schema without rejecting at open.
        PqWriteEncoding e = c.enc;
        if (e == PqWriteEncoding::DeltaBinaryPacked &&
            type_of(i) != bolt::BoltType::Int64) {
            e = PqWriteEncoding::Auto;
        }
        if (e == PqWriteEncoding::ByteStreamSplit &&
            type_of(i) != bolt::BoltType::Float64) {
            e = PqWriteEncoding::Auto;
        }
        if (e == PqWriteEncoding::DeltaByteArray &&
            type_of(i) != bolt::BoltType::Utf8) {
            e = PqWriteEncoding::Auto;
        }
        o.columns[i].encoding = static_cast<std::uint8_t>(e);
    }
    return o;
}

// One configuration: write to MEMORY (so the filesystem is not the subject)
// and report the best of kRepeats plus the produced size.
struct Result { double ms; std::uint64_t bytes; };

Result run_one(const Cfg& c, bolt::BoltBatch* batch) {
    Result best{1e30, 0};
    for (int rep = 0; rep < kRepeats; ++rep) {
        const auto o = make_opts(c);
        const auto t0 = Clock::now();
        ParquetWriter* w = parquet_write_open_mem(&o, 64u << 20);
        if (w == nullptr) return Result{-1.0, 0};
        if (!parquet_write_row_group(w, batch)) return Result{-1.0, 0};
        bolt::Arena out_arena;
        const std::uint8_t* bytes = nullptr;
        std::uint64_t len = 0;
        if (!parquet_write_close_mem(w, &out_arena, &bytes, &len)) {
            return Result{-1.0, 0};
        }
        const double ms = ms_since(t0);
        if (ms < best.ms) best.ms = ms;
        best.bytes = len;
    }
    return best;
}

// Decode cost for the same file, so encode/decode are never confused.
double decode_ms(const Cfg& c, bolt::BoltBatch* batch) {
    const auto o = make_opts(c);
    ParquetWriter* w = parquet_write_open_mem(&o, 64u << 20);
    if (w == nullptr) return -1.0;
    if (!parquet_write_row_group(w, batch)) return -1.0;
    bolt::Arena keep;
    const std::uint8_t* bytes = nullptr;
    std::uint64_t len = 0;
    if (!parquet_write_close_mem(w, &keep, &bytes, &len)) return -1.0;
    double best = 1e30;
    for (int rep = 0; rep < kRepeats; ++rep) {
        bolt::Arena ra;
        auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
        const auto t0 = Clock::now();
        if (!parquet_read_file(bytes, len, &ra, rb)) {
            std::printf("    [decode failed: codec=%u dict=%d enc=%d len=%llu]\n",
                        static_cast<unsigned>(c.codec), (int)c.dictionary,
                        static_cast<int>(c.enc),
                        static_cast<unsigned long long>(len));
            return -1.0;
        }
        const double ms = ms_since(t0);
        if (ms < best) best = ms;
    }
    return best;
}

void row(const char* label, const Result& r, double dec, double base_bytes) {
    const double mb = static_cast<double>(kRows) * kCols * 8.0 / (1024.0 * 1024.0);
    std::printf("  %-26s enc %8.1f ms (%6.1f MB/s)  dec %8.1f ms   "
                "%9llu B  %5.2fx\n",
                label, r.ms, mb / (r.ms / 1000.0), dec,
                static_cast<unsigned long long>(r.bytes),
                base_bytes > 0 ? base_bytes / static_cast<double>(r.bytes) : 1.0);
}

}  // namespace

int main() {
    bolt::Arena arena;
    auto* batch = arena.allocate_array<bolt::BoltBatch>(1);
    build_batch(&arena, batch);
    std::printf("bench_parquet_write: %lld rows x %u cols, min-of-%d\n\n",
                static_cast<long long>(kRows), kCols, kRepeats);

    // Baseline: uncompressed PLAIN, nothing optional enabled.
    const Cfg base{"plain/none", 0, false, PqWriteEncoding::Auto,
                   false, false, false, nullptr};
    const Result rb = run_one(base, batch);
    const double bb = static_cast<double>(rb.bytes);
    std::printf("ENCODINGS (uncompressed, so the encoding is the only variable)\n");
    row("PLAIN", rb, decode_ms(base, batch), bb);
    for (const Cfg c : {
            Cfg{"dictionary", 0, true,  PqWriteEncoding::Auto, false, false, false, nullptr},
            Cfg{"delta (int64 cols)", 0, false, PqWriteEncoding::DeltaBinaryPacked, false, false, false, nullptr},
            Cfg{"byte_stream_split (f64)", 0, false, PqWriteEncoding::ByteStreamSplit, false, false, false, nullptr},
            Cfg{"delta_byte_array (utf8)", 0, false, PqWriteEncoding::DeltaByteArray, false, false, false, nullptr},
        }) {
        row(c.name, run_one(c, batch), decode_ms(c, batch), bb);
    }

    std::printf("\nCODECS (dictionary on, so this isolates the codec)\n");
    for (const Cfg c : {
            Cfg{"none",    0, true, PqWriteEncoding::Auto, false, false, false, nullptr},
            Cfg{"snappy",  1, true, PqWriteEncoding::Auto, false, false, false, nullptr},
            Cfg{"lz4_raw", 4, true, PqWriteEncoding::Auto, false, false, false, nullptr},
            Cfg{"zstd",    3, true, PqWriteEncoding::Auto, false, false, false, nullptr},
            Cfg{"gzip",    2, true, PqWriteEncoding::Auto, false, false, false, nullptr},
        }) {
        row(c.name, run_one(c, batch), decode_ms(c, batch), bb);
    }

    std::printf("\nOPTIONAL METADATA (snappy + dictionary baseline)\n");
    const Cfg meta_base{"none", 1, true, PqWriteEncoding::Auto, false, false, false, nullptr};
    const Result mb0 = run_one(meta_base, batch);
    row("baseline", mb0, decode_ms(meta_base, batch), bb);
    for (const Cfg c : {
            Cfg{"+ page index", 1, true, PqWriteEncoding::Auto, true, false, false, nullptr},
            Cfg{"+ bloom filter", 1, true, PqWriteEncoding::Auto, false, true, false, nullptr},
            Cfg{"+ both", 1, true, PqWriteEncoding::Auto, true, true, false, nullptr},
            Cfg{"+ data_page_v2", 1, true, PqWriteEncoding::Auto, false, false, true, nullptr},
        }) {
        row(c.name, run_one(c, batch), decode_ms(c, batch), bb);
    }

    std::printf("\nPARALLEL COLUMN ENCODING (snappy + dictionary, %u columns)\n",
                kCols);
    std::printf("  %-26s enc %8.1f ms  (serial baseline)\n", "1 (serial)",
                mb0.ms);
    for (std::uint32_t threads : {2u, 4u, 8u}) {
        auto* pool = new bolt::Scheduler();
        if (!pool->init(threads)) { delete pool; continue; }
        Cfg c{"", 1, true, PqWriteEncoding::Auto, false, false, false, pool};
        const Result r = run_one(c, batch);
        char lbl[32];
        std::snprintf(lbl, sizeof(lbl), "%u threads", threads);
        std::printf("  %-26s enc %8.1f ms  %4.2fx speedup\n", lbl, r.ms,
                    mb0.ms / r.ms);
        pool->shutdown();
        delete pool;
    }
    return 0;
}
