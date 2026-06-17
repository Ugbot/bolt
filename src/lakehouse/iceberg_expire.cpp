// bolt/lakehouse/iceberg_expire.cpp — W5 expire-snapshots + remove-orphans.
//
// Both entry points live in iceberg_writer.cpp; this TU is the W5 file-layout
// anchor.

#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kExpireTu = 0; }
bool expire_present() noexcept { return kExpireTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
