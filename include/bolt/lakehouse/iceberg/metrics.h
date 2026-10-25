// bolt/lakehouse/iceberg/metrics.h — Iceberg data-file metrics (spec "Manifests" data_file fields + Appendix F).
//
// What an Iceberg writer records about one parquet data file, computed from the file itself: per leaf column the
// bytes on disk (column_sizes), value / null / NaN counts and lower / upper bounds in the binary single-value
// encoding, and the row groups' start offsets (split_offsets). The rules follow Iceberg-Java (1.12), checked against
// the files it writes:
//   - a metrics mode per column: none (nothing, not even the size), counts (sizes and counts), truncate(N) (counts
//     and bounds, string / binary bounds truncated to N code points / bytes), full (untruncated bounds);
//   - a leaf inside a list or map gets only its size;
//   - NaN counts for float and double; their bounds exclude NaN and order -0.0 before +0.0 (no bounds when every
//     value is NaN or null);
//   - a truncated lower bound is the prefix; a truncated upper bound is the prefix with its last code point (byte)
//     incremented, dropping trailing code points (bytes) that cannot be; no upper bound when none can;
//   - decimal bounds are the unscaled value, two's-complement big-endian, minimal bytes; uuid 16 bytes big-endian;
//     fixed[L] its L bytes, never truncated.
// A bound wider than a ColumnStatEntry slot is truncated the same way (still a valid bound), or dropped for types
// that are never truncated. Tiger Style: bounded loops, no allocation outside the caller's arena.

#pragma once

#include <cstdint>

#include "bolt/lakehouse/iceberg/statistics.h"

namespace bolt {
class Arena;
struct BoltBatch;
namespace ingest { namespace parquet { struct PqMeta; } }
}

namespace bolt {
namespace lakehouse {
namespace iceberg {

struct Schema;

enum class MetricsMode : uint8_t { kNone = 0, kCounts = 1, kTruncate = 2, kFull = 3 };

struct MetricsModeSpec {
    MetricsMode mode   = MetricsMode::kTruncate;
    uint32_t    length = 16u;   // truncate(N)
};

static constexpr uint32_t kIcebergDefaultTruncateLength = 16u;      // write.metadata.metrics.default: truncate(16)
static constexpr uint32_t kIcebergDefaultMaxInferredColumns = 100u; // write.metadata.metrics.max-inferred-column-defaults

// Parse a metrics mode as Iceberg-Java's MetricsModes.fromString: none / counts / full / truncate(N) with N > 0,
// case-insensitive. False on anything else (the caller keeps its default).
bool metrics_mode_parse(const char* s, uint32_t len, MetricsModeSpec* out) noexcept;

// One parquet leaf column of a file: its Iceberg field id (<= 0 skips it) and metrics mode.
struct MetricsLeaf {
    int32_t         field_id = 0;
    MetricsModeSpec mode;
};

// The metrics of the parquet file `bytes[0..len)` whose footer is `meta`; `leaves[c]` describes meta->columns[c]
// (n_leaves may be shorter: the rest are skipped). Writes out->n_cols / cols / split_offsets / sort_order_id;
// record_count and file_size_in_bytes stay the caller's. A column whose values cannot be decoded keeps its counts and
// loses only what needs the values (bounds, NaN count). False only for a malformed footer.
bool file_metrics_from_parquet(const uint8_t* bytes, uint64_t len, const ::bolt::ingest::parquet::PqMeta* meta,
                               const MetricsLeaf* leaves, uint32_t n_leaves, FileStats* out) noexcept;

// The same metrics from the batches a writer is about to serialize as one file, for each primitive field of `sch` the batches
// carry by name, all under `mode`: counts, NaN counts and bounds; no column_sizes or split_offsets (they belong to the file).
bool file_metrics_from_batches(const BoltBatch* const* batches, uint32_t n_batches, const Schema* sch, const MetricsModeSpec& mode,
                               FileStats* out) noexcept;

// Truncation (spec "Appendix F", Iceberg-Java UnicodeUtil / BinaryUtil). `utf8` truncates to `units` code points, else
// to `units` bytes; the result also fits `cap` bytes. Lower: the prefix length. Upper: the incremented prefix written
// to out[0..*out_len); false when no upper bound exists (every unit is already the largest).
uint32_t truncate_lower_len(bool utf8, const uint8_t* v, uint32_t n, uint32_t units, uint32_t cap) noexcept;
bool truncate_upper(bool utf8, const uint8_t* v, uint32_t n, uint32_t units, uint8_t* out, uint32_t cap,
                    uint32_t* out_len) noexcept;

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
