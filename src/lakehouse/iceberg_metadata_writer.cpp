// bolt/lakehouse/iceberg_metadata_writer.cpp — W5 metadata.json emitter.
//
// The actual implementation (Buf-based JSON builder, persist_metadata,
// version-hint.text write) lives in iceberg_writer.cpp so it can share the
// TableHandle definition. This TU exists for the W5 file-layout contract +
// to host the public re-emitter so external callers can serialize a
// Metadata POD on demand.

#include "bolt/lakehouse/iceberg/metadata.h"

#include <cassert>

#include "bolt/bolt_arena.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

// Friend symbol: declared here so external callers can JSON-emit a Metadata
// without holding a TableHandle. Implementation lives in iceberg_writer.cpp.
struct NamedRef;
bool metadata_json_emit(const Metadata* m, const NamedRef* refs, uint32_t nr,
                        bool is_view, const char* view_dialect,
                        const char* view_sql, int64_t view_version_id,
                        Arena* a, const uint8_t** out,
                        uint64_t* out_len) noexcept;

// Convenience wrapper — no refs, no view.
bool metadata_json_emit_plain(const Metadata* m, Arena* a,
                              const uint8_t** out, uint64_t* out_len) noexcept {
    assert(m != nullptr && a != nullptr && out != nullptr && out_len != nullptr);
    return metadata_json_emit(m, nullptr, 0u, false, nullptr, nullptr, 0,
                              a, out, out_len);
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
