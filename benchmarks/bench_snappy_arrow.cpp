// Paired Bolt-vs-Arrow Snappy decode benchmark on real Parquet page bodies.
//
// This file is deliberately not registered in CMake: Arrow is an optional
// measurement oracle, never a Bolt runtime or build dependency. Compile it
// manually against an installed Arrow C++ library; see
// docs/research/59-snappy-back-reference-baseline-audit.md.

#include "bolt/bolt_arena.h"
#include "bolt/ingest/bolt_parquet_meta.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_snappy.h"

#include <arrow/util/compression.h>
#include <arrow/util/config.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace {

using bolt::ingest::parquet::PqChunk;
using bolt::ingest::parquet::PqCodec;
using bolt::ingest::parquet::PqMeta;
using bolt::ingest::parquet::TcCursor;

constexpr uint32_t kMaxSelectedColumns = 16;
constexpr uint32_t kMaxCorpusPages = 1u << 18;
constexpr uint32_t kMaxPagesPerChunk = 1u << 16;
constexpr uint64_t kMaxOutputBytes = 64ull << 20;
constexpr uint32_t kGuardBytes = 32;
constexpr uint32_t kMaxPairs = 31;
constexpr uint8_t kGuardValue = 0xA5;

struct FileMap {
    const uint8_t* data;
    uint64_t size;
    int fd;
};

struct PageHeader {
    int32_t type;
    int32_t uncompressed;
    int32_t compressed;
    int32_t rep_len;
    int32_t def_len;
    bool v2;
    bool v2_compressed;
};

struct PageDesc {
    const uint8_t* src;
    uint32_t src_len;
    uint32_t dst_len;
    uint32_t row_group;
    uint16_t column;
    uint8_t page_type;
};

struct Corpus {
    PageDesc* pages;
    uint32_t count;
    uint64_t compressed_bytes;
    uint64_t uncompressed_bytes;
    uint32_t max_output;
};

struct Sample {
    double wall_ms;
    double cpu_ms;
};

struct Options {
    const char* path;
    const char* row_groups;
    uint16_t columns[kMaxSelectedColumns];
    uint32_t n_columns;
    uint32_t pairs;
    bool validate_only;
};

enum class Engine : uint8_t { Bolt, Arrow };

bool map_file(const char* path, FileMap* out) noexcept {
    assert(path != nullptr);
    assert(out != nullptr);
    out->data = nullptr;
    out->size = 0;
    out->fd = -1;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return false;
    }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                   MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return false;
    }
    out->data = static_cast<const uint8_t*>(p);
    out->size = static_cast<uint64_t>(st.st_size);
    out->fd = fd;
    return true;
}

void unmap_file(FileMap* f) noexcept {
    assert(f != nullptr);
    if (f->data != nullptr) {
        (void)munmap(const_cast<uint8_t*>(f->data), static_cast<size_t>(f->size));
    }
    if (f->fd >= 0) (void)close(f->fd);
    f->data = nullptr;
    f->size = 0;
    f->fd = -1;
}

bool parse_v2_header(TcCursor* c, PageHeader* h) noexcept {
    assert(c != nullptr);
    assert(h != nullptr);
    h->v2 = true;
    h->v2_compressed = true;
    int16_t field = 0;
    uint8_t type = 0;
    while (bolt::ingest::parquet::tc_field(c, &field, &type)) {
        int64_t value = 0;
        if (field == 5 || field == 6) {
            if (!bolt::ingest::parquet::tc_zigzag(c, &value)) return false;
            if (value < 0 || value > INT32_MAX) return false;
            if (field == 5) h->def_len = static_cast<int32_t>(value);
            if (field == 6) h->rep_len = static_cast<int32_t>(value);
        } else if (field == 7) {
            if (type != bolt::ingest::parquet::kTcTrue &&
                type != bolt::ingest::parquet::kTcFalse) return false;
            h->v2_compressed = type == bolt::ingest::parquet::kTcTrue;
        } else if (!bolt::ingest::parquet::tc_skip(c, type, 0)) {
            return false;
        }
    }
    return h->def_len >= 0 && h->rep_len >= 0;
}

bool parse_page_header(TcCursor* c, PageHeader* h) noexcept {
    assert(c != nullptr);
    assert(h != nullptr);
    *h = PageHeader{-1, -1, -1, 0, 0, false, true};
    int16_t field = 0;
    uint8_t type = 0;
    while (bolt::ingest::parquet::tc_field(c, &field, &type)) {
        int64_t value = 0;
        if (field >= 1 && field <= 3) {
            if (!bolt::ingest::parquet::tc_zigzag(c, &value)) return false;
            if (value < 0 || value > INT32_MAX) return false;
            if (field == 1) h->type = static_cast<int32_t>(value);
            if (field == 2) h->uncompressed = static_cast<int32_t>(value);
            if (field == 3) h->compressed = static_cast<int32_t>(value);
        } else if (field == 8) {
            if (type != bolt::ingest::parquet::kTcStruct) return false;
            if (!parse_v2_header(c, h)) return false;
        } else if (!bolt::ingest::parquet::tc_skip(c, type, 0)) {
            return false;
        }
    }
    return h->type >= 0 && h->uncompressed >= 0 && h->compressed >= 0;
}

bool add_page(const uint8_t* body, const PageHeader& h, uint32_t row_group,
              uint16_t column, Corpus* corpus) noexcept {
    assert(body != nullptr);
    assert(corpus != nullptr);
    uint32_t prefix = 0;
    if (h.v2) {
        const uint64_t levels = static_cast<uint64_t>(h.rep_len) + h.def_len;
        if (levels > static_cast<uint64_t>(h.compressed) ||
            levels > static_cast<uint64_t>(h.uncompressed)) return false;
        if (!h.v2_compressed) return true;
        prefix = static_cast<uint32_t>(levels);
    }
    const uint32_t src_len = static_cast<uint32_t>(h.compressed) - prefix;
    const uint32_t dst_len = static_cast<uint32_t>(h.uncompressed) - prefix;
    if (src_len == 0 || dst_len == 0 || dst_len > kMaxOutputBytes) return false;
    uint64_t encoded_len = 0;
    if (bolt::ingest::snappy_uncompressed_len(body + prefix, src_len,
                                               &encoded_len) == 0 ||
        encoded_len != dst_len) return false;
    if (corpus->count >= kMaxCorpusPages) return false;
    corpus->pages[corpus->count++] = PageDesc{
        body + prefix, src_len, dst_len, row_group, column,
        static_cast<uint8_t>(h.type)};
    corpus->compressed_bytes += src_len;
    corpus->uncompressed_bytes += dst_len;
    corpus->max_output = std::max(corpus->max_output, dst_len);
    return true;
}

bool add_chunk(const FileMap& file, const PqChunk& chunk, uint32_t row_group,
               uint16_t column, Corpus* corpus) noexcept {
    assert(file.data != nullptr);
    assert(corpus != nullptr);
    if (chunk.codec != PqCodec::Snappy || chunk.total_compressed_size <= 0) {
        return false;
    }
    const int64_t raw_start = chunk.dictionary_page_offset > 0
        ? chunk.dictionary_page_offset : chunk.data_page_offset;
    if (raw_start <= 0) return false;
    const uint64_t start = static_cast<uint64_t>(raw_start);
    const uint64_t span = static_cast<uint64_t>(chunk.total_compressed_size);
    if (start > file.size || span > file.size - start) return false;
    const uint64_t end = start + span;
    uint64_t pos = start;
    for (uint32_t page = 0; page < kMaxPagesPerChunk && pos < end; ++page) {
        TcCursor cursor{file.data + pos, file.data + end};
        PageHeader header{};
        if (!parse_page_header(&cursor, &header)) return false;
        const uint64_t header_len = static_cast<uint64_t>(cursor.p - (file.data + pos));
        if (header_len > end - pos) return false;
        const uint64_t body_pos = pos + header_len;
        if (static_cast<uint64_t>(header.compressed) > end - body_pos) return false;
        if (!add_page(file.data + body_pos, header, row_group, column, corpus)) {
            return false;
        }
        pos = body_pos + static_cast<uint32_t>(header.compressed);
    }
    return pos == end;
}

bool build_corpus(const FileMap& file, const PqMeta& meta, uint32_t row_groups,
                  const uint16_t* columns, uint32_t n_columns,
                  Corpus* corpus) noexcept {
    assert(columns != nullptr);
    assert(corpus != nullptr);
    if (row_groups == 0 || row_groups > meta.n_row_groups) return false;
    if (n_columns == 0 || n_columns > kMaxSelectedColumns) return false;
    for (uint32_t g = 0; g < row_groups; ++g) {
        const auto& rg = meta.row_groups[g];
        if (rg.chunk_count != meta.n_columns) return false;
        for (uint32_t i = 0; i < n_columns; ++i) {
            if (columns[i] >= meta.n_columns) return false;
            const uint32_t chunk_idx = rg.chunk_off + columns[i];
            if (chunk_idx >= meta.n_chunks) return false;
            if (!add_chunk(file, meta.chunks[chunk_idx], g, columns[i], corpus)) {
                std::fprintf(stderr, "page extraction failed at rg=%u col=%u\n",
                             g, static_cast<unsigned>(columns[i]));
                return false;
            }
        }
    }
    return corpus->count > 0 && corpus->max_output > 0;
}

uint64_t fnv_bytes(uint64_t h, const uint8_t* p, uint32_t n) noexcept {
    assert(p != nullptr || n == 0);
    for (uint32_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

bool decode_page(Engine engine, arrow::util::Codec* arrow_codec,
                 const PageDesc& page, uint8_t* out) noexcept {
    assert(arrow_codec != nullptr);
    assert(out != nullptr);
    if (engine == Engine::Bolt) {
        return bolt::ingest::snappy_decompress(page.src, page.src_len,
                                                out, page.dst_len);
    }
    auto result = arrow_codec->Decompress(page.src_len, page.src,
                                          page.dst_len, out);
    return result.ok() && result.ValueUnsafe() == page.dst_len;
}

bool guard_intact(const uint8_t* out, uint32_t n) noexcept {
    assert(out != nullptr);
    for (uint32_t i = 0; i < kGuardBytes; ++i) {
        if (out[n + i] != kGuardValue) return false;
    }
    return true;
}

bool validate_corpus(const Corpus& corpus, arrow::util::Codec* arrow_codec,
                     uint8_t* bolt_out, uint8_t* arrow_out,
                     uint64_t* checksum) noexcept {
    assert(corpus.pages != nullptr);
    assert(checksum != nullptr);
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < corpus.count; ++i) {
        const PageDesc& page = corpus.pages[i];
        std::memset(bolt_out + page.dst_len, kGuardValue, kGuardBytes);
        std::memset(arrow_out + page.dst_len, kGuardValue, kGuardBytes);
        if (!decode_page(Engine::Bolt, arrow_codec, page, bolt_out) ||
            !decode_page(Engine::Arrow, arrow_codec, page, arrow_out) ||
            !guard_intact(bolt_out, page.dst_len) ||
            !guard_intact(arrow_out, page.dst_len) ||
            std::memcmp(bolt_out, arrow_out, page.dst_len) != 0) {
            std::fprintf(stderr, "mismatch at page=%u rg=%u col=%u type=%u\n",
                         i, page.row_group, static_cast<unsigned>(page.column),
                         static_cast<unsigned>(page.page_type));
            return false;
        }
        hash = fnv_bytes(hash, reinterpret_cast<const uint8_t*>(&page.dst_len),
                         sizeof(page.dst_len));
        hash = fnv_bytes(hash, bolt_out, page.dst_len);
    }
    *checksum = hash;
    return true;
}

double timespec_ms(const timespec& t) noexcept {
    return static_cast<double>(t.tv_sec) * 1000.0 +
           static_cast<double>(t.tv_nsec) / 1.0e6;
}

bool run_sweep(Engine engine, const Corpus& corpus,
               arrow::util::Codec* arrow_codec, uint8_t* out,
               Sample* sample) noexcept {
    assert(out != nullptr);
    assert(sample != nullptr);
    timespec cpu_start{};
    timespec cpu_end{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start) != 0) return false;
    const auto wall_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < corpus.count; ++i) {
        if (!decode_page(engine, arrow_codec, corpus.pages[i], out)) return false;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) return false;
    sample->wall_ms = std::chrono::duration<double, std::milli>(
        wall_end - wall_start).count();
    sample->cpu_ms = timespec_ms(cpu_end) - timespec_ms(cpu_start);
    return sample->wall_ms > 0.0 && sample->cpu_ms > 0.0;
}

double median(double* values, uint32_t n) noexcept {
    assert(values != nullptr);
    assert(n > 0 && n <= kMaxPairs);
    std::sort(values, values + n);
    if ((n & 1u) != 0u) return values[n / 2u];
    return (values[n / 2u - 1u] + values[n / 2u]) * 0.5;
}

bool run_pairs(const Corpus& corpus, arrow::util::Codec* codec, uint8_t* out,
               uint32_t pairs) noexcept {
    assert(pairs > 0 && pairs <= kMaxPairs);
    Sample bolt_samples[kMaxPairs]{};
    Sample arrow_samples[kMaxPairs]{};
    for (uint32_t p = 0; p < pairs; ++p) {
        const bool bolt_first = (p & 1u) == 0u;
        Sample* first = bolt_first ? &bolt_samples[p] : &arrow_samples[p];
        Sample* second = bolt_first ? &arrow_samples[p] : &bolt_samples[p];
        const Engine first_engine = bolt_first ? Engine::Bolt : Engine::Arrow;
        const Engine second_engine = bolt_first ? Engine::Arrow : Engine::Bolt;
        if (!run_sweep(first_engine, corpus, codec, out, first) ||
            !run_sweep(second_engine, corpus, codec, out, second)) return false;
        std::printf("pair %u order=%s bolt_cpu=%.3f arrow_cpu=%.3f "
                    "bolt_wall=%.3f arrow_wall=%.3f ratio_cpu=%.4f\n",
                    p, bolt_first ? "BA" : "AB", bolt_samples[p].cpu_ms,
                    arrow_samples[p].cpu_ms, bolt_samples[p].wall_ms,
                    arrow_samples[p].wall_ms,
                    bolt_samples[p].cpu_ms / arrow_samples[p].cpu_ms);
    }
    double cpu_ratios[kMaxPairs]{};
    double wall_ratios[kMaxPairs]{};
    for (uint32_t p = 0; p < pairs; ++p) {
        cpu_ratios[p] = bolt_samples[p].cpu_ms / arrow_samples[p].cpu_ms;
        wall_ratios[p] = bolt_samples[p].wall_ms / arrow_samples[p].wall_ms;
    }
    const double cpu_median = median(cpu_ratios, pairs);
    const double wall_median = median(wall_ratios, pairs);
    std::printf("summary pairs=%u median_arrow_speedup_cpu=%.4f "
                "median_arrow_speedup_wall=%.4f\n", pairs, cpu_median,
                wall_median);
    return true;
}

bool parse_columns(const char* text, uint16_t* columns,
                   uint32_t* n_columns) noexcept {
    assert(text != nullptr);
    assert(columns != nullptr && n_columns != nullptr);
    const char* p = text;
    uint32_t count = 0;
    while (*p != '\0' && count < kMaxSelectedColumns) {
        char* end = nullptr;
        errno = 0;
        const unsigned long value = std::strtoul(p, &end, 10);
        if (errno != 0 || end == p || value > UINT16_MAX) return false;
        columns[count++] = static_cast<uint16_t>(value);
        if (*end == '\0') {
            p = end;
        } else if (*end == ',') {
            p = end + 1;
        } else {
            return false;
        }
    }
    if (*p != '\0' || count == 0) return false;
    *n_columns = count;
    return true;
}

uint32_t parse_count(const char* text, uint32_t maximum) noexcept {
    assert(text != nullptr);
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > maximum) return 0;
    return static_cast<uint32_t>(value);
}

void usage(const char* argv0) noexcept {
    std::fprintf(stderr,
        "usage: %s FILE ROW_GROUPS|all COLS_CSV [PAIRS|--validate-only]\n",
        argv0);
}

bool parse_options(int argc, char** argv, Options* options) noexcept {
    assert(argv != nullptr);
    assert(options != nullptr);
    if (argc < 4 || argc > 5) return false;
    options->path = argv[1];
    options->row_groups = argv[2];
    options->pairs = 9;
    options->validate_only = false;
    if (!parse_columns(argv[3], options->columns, &options->n_columns)) {
        return false;
    }
    if (argc == 5) {
        if (std::strcmp(argv[4], "--validate-only") == 0) {
            options->validate_only = true;
        } else {
            options->pairs = parse_count(argv[4], kMaxPairs);
            if (options->pairs == 0) return false;
        }
    }
    return true;
}

bool prepare_corpus(const Options& options, const FileMap& file,
                    bolt::Arena* meta_arena, Corpus* corpus) noexcept {
    assert(meta_arena != nullptr);
    assert(corpus != nullptr);
    PqMeta meta{};
    if (!bolt::ingest::parquet::parquet_read_meta(file.data, file.size,
                                                  meta_arena, &meta)) {
        std::fprintf(stderr, "parquet metadata parse failed\n");
        return false;
    }
    const uint32_t row_groups = std::strcmp(options.row_groups, "all") == 0
        ? meta.n_row_groups : parse_count(options.row_groups, meta.n_row_groups);
    if (row_groups == 0) return false;
    corpus->pages = static_cast<PageDesc*>(
        std::calloc(kMaxCorpusPages, sizeof(PageDesc)));
    if (corpus->pages == nullptr) return false;
    if (!build_corpus(file, meta, row_groups, options.columns,
                      options.n_columns, corpus)) {
        std::free(corpus->pages);
        corpus->pages = nullptr;
        return false;
    }
    return true;
}

int execute_corpus(const Options& options, const Corpus& corpus) noexcept {
    auto codec_result = arrow::util::Codec::Create(arrow::Compression::SNAPPY);
    if (!codec_result.ok()) {
        std::fprintf(stderr, "Arrow Snappy codec unavailable\n");
        return 2;
    }
    std::unique_ptr<arrow::util::Codec> codec =
        std::move(codec_result).ValueUnsafe();
    const size_t cap = static_cast<size_t>(corpus.max_output) + kGuardBytes;
    auto* bolt_out = static_cast<uint8_t*>(std::malloc(cap));
    auto* arrow_out = static_cast<uint8_t*>(std::malloc(cap));
    if (bolt_out == nullptr || arrow_out == nullptr) {
        std::free(bolt_out);
        std::free(arrow_out);
        return 2;
    }
    uint64_t checksum = 0;
    const bool valid = validate_corpus(corpus, codec.get(), bolt_out,
                                       arrow_out, &checksum);
    if (valid) {
        std::printf("validated pages=%u compressed=%llu uncompressed=%llu "
                    "max_page=%u checksum=%016llx arrow=%s compiler=%s\n",
                    corpus.count,
                    static_cast<unsigned long long>(corpus.compressed_bytes),
                    static_cast<unsigned long long>(corpus.uncompressed_bytes),
                    corpus.max_output, static_cast<unsigned long long>(checksum),
                    ARROW_VERSION_STRING, __clang_version__);
    }
    bool measured = true;
    if (valid && !options.validate_only) {
        Sample warm_bolt{};
        Sample warm_arrow{};
        measured = run_sweep(Engine::Bolt, corpus, codec.get(), bolt_out,
                             &warm_bolt) &&
                   run_sweep(Engine::Arrow, corpus, codec.get(), arrow_out,
                             &warm_arrow) &&
                   run_pairs(corpus, codec.get(), bolt_out, options.pairs);
    }
    std::free(bolt_out);
    std::free(arrow_out);
    return valid && measured ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    FileMap file{};
    if (!map_file(options.path, &file)) {
        std::fprintf(stderr, "cannot mmap %s: %s\n", options.path,
                     std::strerror(errno));
        return 2;
    }
    bolt::ArenaConfig meta_config;
    meta_config.initial_block_size = 16ull << 20;
    meta_config.max_block_size = 256ull << 20;
    bolt::Arena meta_arena(meta_config);
    Corpus corpus{};
    if (!prepare_corpus(options, file, &meta_arena, &corpus)) {
        std::fprintf(stderr, "failed to build bounded page corpus\n");
        unmap_file(&file);
        return 2;
    }
    const int result = execute_corpus(options, corpus);
    std::free(corpus.pages);
    unmap_file(&file);
    return result;
}
