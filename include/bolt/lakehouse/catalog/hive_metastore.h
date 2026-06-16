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

// --- Thrift Binary Protocol codec (exposed for tests) ---

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
