// bolt/lakehouse/iceberg_compaction.cpp — W5 compaction / rewrite path.
//
// table_rewrite + table_compact entry points live in iceberg_writer.cpp;
// they emit a kReplace snapshot. Full rewrite-small-files logic (the
// upstream design's "files < target/2 rewrite") is the next slice.

#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kCompactionTu = 0; }
bool compaction_present() noexcept { return kCompactionTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
