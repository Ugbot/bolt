// Dev/oracle tool: scan an Iceberg table in a filesystem catalog with bolt's own reader and print its first (int64) column as JSON.
//   bolt_iceberg_scan_dump <catalog_root> <namespace> <table>
// Used by scripts/oracles/verify_v3_external.py to compare bolt against Iceberg-Java on tables bolt did not write.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/lakehouse/catalog.h"
#include "bolt/lakehouse/iceberg/metadata.h"
#include "bolt/lakehouse/iceberg/scan.h"

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s <catalog_root> <ns> <table>\n", argv[0]); return 2; }
    using namespace bolt::lakehouse;
    using namespace bolt::lakehouse::iceberg;
    bolt::Arena arena;
    FilesystemCatalog fc{}; Catalog cat{};
    if (!filesystem_catalog_init(&fc, argv[1], &cat)) { std::fprintf(stderr, "catalog init failed\n"); return 3; }
    TableHandle* th = nullptr;
    if (!iceberg_table_open(&th, &arena, &cat, argv[2], argv[3])) { std::fprintf(stderr, "table open failed\n"); return 4; }
    ReadOptions ro{}; read_options_init(&ro);
    ScanHandle* sh = nullptr;
    if (!iceberg_scan_open(&sh, th, &ro)) { std::fprintf(stderr, "scan open failed\n"); return 5; }
    std::printf("{\"ids\":[");
    bool first = true;
    for (int g = 0; g < 100000; ++g) {
        bolt::BoltBatch b{}; bool eof = false;
        if (!iceberg_scan_next_batch(sh, &b, &eof)) { std::fprintf(stderr, "scan failed\n"); return 6; }
        if (eof) break;
        const auto& c = b.columns[b.read_epoch][0];
        const auto* v = static_cast<const int64_t*>(c.data);
        for (int64_t r = 0; r < b.num_rows; ++r) { std::printf("%s%lld", first ? "" : ",", static_cast<long long>(v[r])); first = false; }
    }
    std::printf("]}\n");
    iceberg_scan_close(sh); iceberg_table_close(th);
    return 0;
}
