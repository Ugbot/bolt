// bench_wire_serialize.cpp — G2ICE-141: measure bolt_wire_serialize on the
// realistic ingest batch shape from the ticket's dtrace finding:
// 1024 rows x {ts:int64, pk:uint64, emb:vector<f32,768>} ~= 3.1 MB/batch.
// The pre-fix serializer memset the ENTIRE wire buffer before packing
// (~29% of the synchronous ingest hot path); this bench exists to measure
// that share before/after the narrowed zero-fill.
//
// Usage: bench_wire_serialize [seconds=10] [rows=1024] [dim=768]

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"
#include "bolt/wire/bolt_wire.h"

using namespace bolt;

int main(int argc, char** argv) {
    const double seconds = (argc > 1) ? atof(argv[1]) : 10.0;
    const int64_t rows   = (argc > 2) ? atoll(argv[2]) : 1024;
    const uint32_t dim   = (argc > 3) ? static_cast<uint32_t>(atoi(argv[3])) : 768u;

    Arena arena;
    BoltBatch b;
    BoltBatch::init_empty(&b);
    b.arena = &arena; b.num_cols = 3; b.num_rows = rows;
    b.schema.num_fields = 3;
    BoltBatch::alloc_columns(&b, &arena, 3);

    auto set_field = [&](uint32_t i, const char* name, BoltType t, uint32_t fs) {
        BoltField& f = b.schema.fields[i];
        memset(&f, 0, sizeof(f));
        f.set_name(name); f.type = t; f.nullable = false; f.fixed_size = fs;
    };
    set_field(0, "ts",  BoltType::Int64,  0);
    set_field(1, "pk",  BoltType::UInt64, 0);
    set_field(2, "emb", BoltType::Embedding, dim);

    BoltColumn c0 = BoltColumn::make_flat_alloc(rows, BoltType::Int64, &arena);
    BoltColumn c1 = BoltColumn::make_flat_alloc(rows, BoltType::UInt64, &arena);
    BoltColumn c2 = BoltColumn::make_flat_vector_alloc(rows, dim, &arena);
    if (!c0.data || !c1.data || !c2.data) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int64_t r = 0; r < rows; ++r) {
        static_cast<int64_t*>(c0.data)[r]  = 1'700'000'000'000'000'000LL + r;
        static_cast<uint64_t*>(c1.data)[r] = 0x9E3779B97F4A7C15ull * (r + 1);
    }
    auto* emb = static_cast<float*>(c2.data);
    for (int64_t i = 0; i < rows * dim; ++i)
        emb[i] = static_cast<float>((i * 2654435761u) % 1000) * 0.001f;
    b.columns[0][0] = c0; b.columns[1][0] = c0;
    b.columns[0][1] = c1; b.columns[1][1] = c1;
    b.columns[0][2] = c2; b.columns[1][2] = c2;

    const size_t need = wire::bolt_wire_size(&b);
    if (need == 0) { fprintf(stderr, "bolt_wire_size failed\n"); return 1; }
    uint8_t* buf = static_cast<uint8_t*>(arena.allocate(need, 64));
    if (!buf) { fprintf(stderr, "buf alloc failed\n"); return 1; }

    printf("batch: %lld rows, dim %u, wire size %zu bytes (%.2f MB)\n",
           (long long)rows, dim, need, need / 1048576.0);

    // Warmup + checksum sink to defeat DCE.
    uint64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t iters = 0;
    for (;;) {
        const size_t w = wire::bolt_wire_serialize(&b, buf, need);
        if (w != need) { fprintf(stderr, "serialize failed\n"); return 1; }
        sink += buf[(iters * 4097) % need];
        ++iters;
        if ((iters & 15u) == 0) {
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (el >= seconds) break;
        }
    }
    const double el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double gbs = (static_cast<double>(iters) * need) / el / 1e9;
    printf("iters=%" PRIu64 "  elapsed=%.3fs  %.1f serialize/s  %.2f GB/s  (sink %" PRIu64 ")\n",
           iters, el, iters / el, gbs, sink);
    return 0;
}
