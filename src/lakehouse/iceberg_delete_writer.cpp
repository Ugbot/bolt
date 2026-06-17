// bolt/lakehouse/iceberg_delete_writer.cpp — W5 equality + position delete
// writers.
//
// table_delete in iceberg_writer.cpp emits an empty kDelete snapshot today.
// Full equality/position delete-file emission (Parquet body with
// equality_ids[] / (file_path, pos) columns, manifest content=1/2) is the
// next slice — gated by an Iceberg v2 round-trip test against the read path's
// delete consumer.

#include "bolt/lakehouse/iceberg/delete_file.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kDeleteWriterTu = 0; }
bool delete_writer_present() noexcept { return kDeleteWriterTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
