// bolt/lakehouse/catalog/hive_metastore.h — Hive Metastore Thrift catalog.
// Plain TCP to host:9083, Thrift Binary Protocol. No TLS.
#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/catalog.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace hive_metastore {

struct Config {
    char     host[256];
    uint16_t port;            // typically 9083
};

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept;

// --- High-level surface (G2CLUS-179) ---
//
// get_table(db, name) → the table's storage-descriptor location, copied into
// out_loc[0..cap). list_namespaces/list_tables are on the CatalogVT.
int hms_get_table(Catalog* cat, const char* db, const char* table,
                  char* out_loc, uint32_t cap) noexcept;

// alter_table(db, name, new_location) — rewrites the table's sd.location.
int hms_alter_table_location(Catalog* cat, const char* db, const char* table,
                             const char* new_location) noexcept;

// add_partitions(db, table, values[][n_cols], n_parts, n_cols, location_fmt) —
// adds `n_parts` partitions, each a list of `n_cols` string values, sharing the
// base `location` (Hive appends the partition path). Returns kCat* code.
int hms_add_partition(Catalog* cat, const char* db, const char* table,
                      const char* const* values, uint32_t n_cols,
                      const char* location) noexcept;

// drop_partition(db, table, values[n_cols]) — drops one partition by its
// partition-column values.
int hms_drop_partition(Catalog* cat, const char* db, const char* table,
                       const char* const* values, uint32_t n_cols,
                       bool delete_data) noexcept;

// --- Thrift Binary Protocol codec (exposed for tests) ---

// Encode an i32 field (id, value) into buf at *cursor. Returns false on overflow.
bool hms_encode_i32_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                          int16_t fid, int32_t value) noexcept;

// Encode a bool field. Returns false on overflow.
bool hms_encode_bool_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                           int16_t fid, bool value) noexcept;

// Encode a list<string> field (id, items, n) into buf at *cursor. Returns
// false on overflow.
bool hms_encode_string_list_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                                  int16_t fid, const char* const* items,
                                  uint32_t n) noexcept;

// Build a get_table CALL argument struct (db, table) into buf. Returns length
// (including STOP) or 0 on overflow.
uint32_t hms_build_get_table_args(uint8_t* buf, uint32_t cap, const char* db,
                                  const char* table) noexcept;

// Build a drop_partition CALL argument struct (db, table, part-vals list,
// deleteData) into buf. Returns length or 0 on overflow.
uint32_t hms_build_drop_partition_args(uint8_t* buf, uint32_t cap,
                                       const char* db, const char* table,
                                       const char* const* values,
                                       uint32_t n_cols,
                                       bool delete_data) noexcept;

// Encode a string field (id, bytes, len) into buf at *cursor.
// Returns false on overflow.
bool hms_encode_string_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                             int16_t fid, const char* s, uint32_t n) noexcept;

// Encode a STOP byte (0).
bool hms_encode_struct_stop(uint8_t* buf, uint32_t cap, uint32_t* cursor) noexcept;

// Decode a string at buf[cursor]: reads i32 length + bytes. NUL-terminates `out`.
// Returns false on overflow.
bool hms_decode_string(const uint8_t* buf, uint32_t len, uint32_t* cursor,
                       char* out, uint32_t cap) noexcept;

// Decode a field header. *out_type=0 means STOP. Advances cursor.
bool hms_decode_field_header(const uint8_t* buf, uint32_t len, uint32_t* cursor,
                             uint8_t* out_type, int16_t* out_fid) noexcept;

}  // namespace hive_metastore
}  // namespace lakehouse
}  // namespace bolt
