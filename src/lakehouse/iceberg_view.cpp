// bolt/lakehouse/iceberg_view.cpp — W5 view create + replace.
//
// view_create + view_replace live in iceberg_writer.cpp.

#include "bolt/lakehouse/iceberg/view.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kViewTu = 0; }
bool view_tu_present() noexcept { return kViewTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
