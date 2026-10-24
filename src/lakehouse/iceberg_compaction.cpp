// bolt/lakehouse/iceberg_compaction.cpp — W5 compaction / rewrite path.
//
// table_rewrite + table_compact live in iceberg_compact.inc (included by
// iceberg_writer.cpp, G2ICE-280); the rows they rewrite are read through
// iceberg_dml_read.cpp.

#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kCompactionTu = 0; }
bool compaction_present() noexcept { return kCompactionTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
