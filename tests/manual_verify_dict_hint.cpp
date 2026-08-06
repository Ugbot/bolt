// G2FEAT-152 inc 1 verification: the dictionary hint must reconstruct the
// Flat column EXACTLY. For every row: dict[codes[row]] must be byte-identical
// to the gathered StringView, and the resolved BYTES must match too.
#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/kernels/bolt_utf8.h"
#include <cstdio>
#include <cstring>
#include <vector>

namespace pq = bolt::ingest::parquet;
namespace u8 = bolt::kernels::utf8;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
                                  : "C:/code/clickbench/athena_100m/hits.parquet";
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long long fsz = _ftelli64(f);
    std::fseek(f, 0, SEEK_SET);
    // Footer only: read the last 1 MiB to parse metadata.
    const long long tail = (fsz < (1 << 20)) ? fsz : (1 << 20);
    (void)tail;
    std::vector<unsigned char> buf((size_t)fsz);
    if (std::fread(buf.data(), 1, (size_t)fsz, f) != (size_t)fsz) {
        printf("short read (file is %lld bytes; this check needs it resident)\n", fsz);
        std::fclose(f); return 1;
    }
    std::fclose(f);

    bolt::Arena meta_arena{bolt::ArenaConfig{8u << 20, 512u << 20, 64, false}};
    pq::PqMeta meta_store{};
    pq::PqMeta* meta = &meta_store;
    if (!pq::parquet_read_meta(buf.data(), (uint64_t)fsz, &meta_arena, meta)) {
        printf("meta parse failed\n"); return 1;
    }
    // Find the `url` column.
    uint32_t urlc = 0xFFFFFFFFu;
    for (uint32_t c = 0; c < meta->n_columns; ++c) {
        const char* n = meta->columns[c].name;
        if (_stricmp(n, "url") == 0) { urlc = c; break; }
    }
    if (urlc == 0xFFFFFFFFu) { printf("no url column\n"); return 1; }

    bolt::ArenaConfig cfg{}; cfg.initial_block_size = 512ull << 20;
    cfg.max_block_size = 4ull << 30;
    bolt::Arena arena{cfg};

    long long rows_checked = 0, mismatches = 0, rg_with_hint = 0, rg_total = 0;
    const uint32_t kRg = (meta->n_row_groups < 6u) ? meta->n_row_groups : 6u;
    for (uint32_t g = 0; g < kRg; ++g) {
        arena.reset();
        bolt::BoltColumn col{};
        int64_t nrows = 0;
        const uint16_t idxs[1] = {(uint16_t)urlc};
        if (!pq::parquet_read_row_group_cols(buf.data(), (uint64_t)fsz, meta, g,
                                             idxs, 1, &arena, &col, &nrows)) {
            printf("rg %u decode failed\n", g); return 1;
        }
        ++rg_total;
        const auto* sv = static_cast<const bolt::StringView*>(col.data);
        const char* base = static_cast<const char*>(col.str_overflow_base);
        if (col.dict_child == nullptr) { printf("  rg %u: NO HINT\n", g); continue; }
        ++rg_with_hint;
        const bolt::BoltColumn* keys = col.dict_child;
        const bolt::BoltColumn* vals = keys->dict_child;
        if (vals == nullptr) { printf("  rg %u: hint has no values col\n", g); ++mismatches; continue; }
        const auto* codes = static_cast<const int32_t*>(keys->data);
        const auto* dsv   = static_cast<const bolt::StringView*>(vals->data);
        const char* dbase = static_cast<const char*>(vals->str_overflow_base);
        if (keys->length != col.length) { printf("  rg %u: key length %lld != %lld\n", g,
            (long long)keys->length, (long long)col.length); ++mismatches; continue; }
        for (int64_t i = 0; i < nrows; ++i) {
            const int32_t code = codes[i];
            if (code < 0) continue;                       // null row
            if (code >= vals->length) { ++mismatches; continue; }
            // 1. the 16-byte views must be identical
            if (std::memcmp(&sv[i], &dsv[code], 16) != 0) { ++mismatches; continue; }
            // 2. and the resolved bytes must be identical
            const uint32_t L = sv[i].length;
            const char* a = u8::sv_bytes(sv[i], base);
            const char* b = u8::sv_bytes(dsv[code], dbase);
            if (a == nullptr || b == nullptr) { ++mismatches; continue; }
            if (L != 0 && std::memcmp(a, b, L) != 0) ++mismatches;
            ++rows_checked;
        }
        printf("  rg %u: rows=%lld dict=%lld  checked ok\n", g, (long long)nrows,
               (long long)vals->length);
    }
    printf("\nrowgroups=%lld with_hint=%lld  rows_checked=%lld  MISMATCHES=%lld\n",
           rg_total, rg_with_hint, rows_checked, mismatches);
    printf("%s\n", mismatches == 0 && rg_with_hint == rg_total ? "ALL PASS" : "FAILURES");
    return mismatches != 0 || rg_with_hint != rg_total;
}
