// Single-threaded, single-column parquet decode microbenchmark.
//
// Why this exists: the end-to-end query `SELECT sum(regionid) FROM hits` spends
// ~40% of its thread time in scheduler idle/spin, and this box runs several
// build+benchmark agents at once, so query wall time has a ~40% run-to-run
// spread — far too noisy to attribute a decode change. This driver calls
// bolt::ingest::parquet::parquet_read_row_group_cols directly on one column of
// N row groups on ONE thread, so it measures decode and nothing else.
//
// It also prints a checksum of the decoded values so a "faster" decoder that
// changed a single value is caught immediately.
//
//   bench_pq_decode_col <file.parquet> <col_index> [n_row_groups] [reps]

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_read.h"

namespace {

// Sum the decoded column as raw integers, whatever its width. Not a SQL SUM:
// a wrapping accumulation is fine and is exactly what we want as a
// bit-for-bit fingerprint of the decoded buffer.
uint64_t checksum_column(const bolt::BoltColumn& c, int64_t rows) noexcept {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = static_cast<const uint8_t*>(c.data);
    if (p == nullptr) return 0;
    const uint32_t w = static_cast<uint32_t>(bolt::type_size(c.type));
    for (int64_t i = 0; i < rows; ++i) {
        uint64_t v = 0;
        std::memcpy(&v, p + static_cast<uint64_t>(i) * w, w > 8u ? 8u : w);
        h = (h ^ v) * 1099511628211ull;
    }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <file.parquet> <col_index> [n_rg] [reps]\n",
                     argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const uint32_t col = static_cast<uint32_t>(std::atoi(argv[2]));
    const uint32_t n_rg_want = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3])) : 16u;
    const uint32_t reps = (argc > 4) ? static_cast<uint32_t>(std::atoi(argv[4])) : 5u;

    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) { std::perror("open"); return 1; }
    struct stat st;
    if (::fstat(fd, &st) != 0) { std::perror("fstat"); return 1; }
    const uint64_t len = static_cast<uint64_t>(st.st_size);
    void* m = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { std::perror("mmap"); return 1; }
    const uint8_t* buf = static_cast<const uint8_t*>(m);

    // Meta arena is separate and never reset; the decode arena is reset per
    // row group so the measurement excludes allocator growth.
    bolt::Arena meta_arena;
    bolt::ingest::parquet::PqMeta meta;
    if (!bolt::ingest::parquet::parquet_read_meta(buf, len, &meta_arena, &meta)) {
        std::fprintf(stderr, "parquet_read_meta failed\n");
        return 1;
    }
    if (col >= meta.n_columns) {
        std::fprintf(stderr, "col %u out of range (%u cols)\n", col, meta.n_columns);
        return 1;
    }
    const uint32_t n_rg = (n_rg_want < meta.n_row_groups) ? n_rg_want : meta.n_row_groups;

    bolt::Arena arena;

    // Warm the page cache for the chunks we will touch, so we time decode and
    // not first-touch page faults.
    uint64_t bytes = 0;
    const uint16_t cidx = static_cast<uint16_t>(col);
    for (uint32_t rep = 0; rep < reps + 1u; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t sum_rows = 0;
        uint64_t hash = 1469598103934665603ull;
        bytes = 0;
        for (uint32_t rg = 0; rg < n_rg; ++rg) {
            arena.reset();
            bolt::BoltColumn out{};
            int64_t rows = 0;
            if (!bolt::ingest::parquet::parquet_read_row_group_cols(
                    buf, len, &meta, rg, &cidx, 1u, &arena, &out, &rows)) {
                std::fprintf(stderr, "decode failed at rg %u\n", rg);
                return 1;
            }
            sum_rows += static_cast<uint64_t>(rows);
            hash ^= checksum_column(out, rows);
            hash *= 1099511628211ull;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (rep == 0) {
            std::printf("warmup            rows=%" PRIu64 " ck=%016" PRIx64 "\n",
                        sum_rows, hash);
            continue;
        }
        std::printf("rep %u  %8.2f ms  rows=%" PRIu64 "  %6.1f Mrow/s  ck=%016" PRIx64 "\n",
                    rep, ms, sum_rows,
                    static_cast<double>(sum_rows) / (ms * 1000.0), hash);
        (void)bytes;
    }
    ::munmap(m, len);
    ::close(fd);
    return 0;
}
