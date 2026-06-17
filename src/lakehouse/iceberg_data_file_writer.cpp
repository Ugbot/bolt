// bolt/lakehouse/iceberg_data_file_writer.cpp — W5 Parquet data-file writer.
//
// The actual Parquet emit lives inside iceberg_writer.cpp::write_data_file —
// it calls bolt::ingest::parquet::parquet_write_open/row_group/close. This
// TU exists for the W5 file-layout contract.

#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/writer.h"

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace { volatile int kDataFileWriterTu = 0; }
bool data_file_writer_present() noexcept { return kDataFileWriterTu == 0; }

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
