// bolt/lakehouse/iceberg_manifest_writer.cpp — W5 manifest + manifest-list
// emitters. JSON form (matches the read path's BOLT_ICEBERG_MANIFEST_JSON).
//
// The actual emitters (emit_manifest_json / emit_manifest_list_json) are
// hosted inside iceberg_writer.cpp's anonymous namespace because they are
// called from there. Avro-encoded v2 manifests remain a TODO — see
// docs/research/bolt-lakehouse-design.md.

#include "bolt/lakehouse/iceberg/manifest.h"

#include <cstdint>

namespace bolt {
namespace lakehouse {
namespace iceberg {

// Stub anchor — keeps the TU non-empty for MSVC's "no symbols" warning.
namespace {
volatile int kManifestWriterTu = 0;
}

bool manifest_writer_present() noexcept { return kManifestWriterTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
