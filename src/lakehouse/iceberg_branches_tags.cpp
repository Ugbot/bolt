// bolt/lakehouse/iceberg_branches_tags.cpp — W5 branch + tag refs.
//
// table_create_branch / table_create_tag / table_drop_ref live in
// iceberg_writer.cpp and use a bounded NamedRef[16] table inside TableHandle.

#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kBranchesTagsTu = 0; }
bool branches_tags_present() noexcept { return kBranchesTagsTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
