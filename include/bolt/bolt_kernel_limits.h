// bolt_kernel_limits.h: caps of single kernels, parsers and formats, in the namespace of the code they bound. Each is fixed by a format, a
// foreign implementation it matches, or a fixed-size scratch: kind A invariants, not budgets, so they are not in the registry of bolt_limits.h.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bolt::cal {
// bolt_calendar.h
inline constexpr std::int64_t kMaxAbsDays = 106751991LL;
}  // namespace bolt::cal

namespace bolt::dtfmt {
// bolt_datetime_fmt.h
inline constexpr std::uint32_t kMaxFormattedBytes = 256;
inline constexpr std::uint32_t kMaxLiteralBytes = 128;
inline constexpr std::uint32_t kMaxPatternBytes = 128;
inline constexpr std::uint32_t kMaxTokens = 64;
}  // namespace bolt::dtfmt

namespace bolt::kernels::hll {
// bolt_hll.h
inline constexpr std::uint32_t k_hll_rt_max_p = 26;
// bolt_hll_spark.h
inline constexpr std::uint32_t k_hll_spark_bias_max_p = 18;
}  // namespace bolt::kernels::hll

namespace bolt::kernels::jsonspark {
// bolt_json_spark.h
inline constexpr std::uint32_t kMaxEvalDepth = 256;      // path-evaluator recursion (flattened array nesting); past it: budget refusal
inline constexpr std::uint32_t kMaxNesting = 1000;       // Jackson StreamReadConstraints.maxNestingDepth
inline constexpr std::uint32_t kMaxNumberChars = 1000;   // StreamReadConstraints.maxNumberLength
inline constexpr std::uint32_t kMaxPathBytes = 4096;     // bytes of names in one compiled path
inline constexpr std::uint32_t kMaxPathInstr = 256;      // compiled path instructions (budget json_path_instructions)
}  // namespace bolt::kernels::jsonspark

namespace bolt::kernels::nested_gather_detail {
// bolt_nested_gather.h
inline constexpr std::uint32_t kMaxChildren = 65536;
inline constexpr std::uint32_t kMaxDepth = 32;
}  // namespace bolt::kernels::nested_gather_detail

namespace bolt::kernels::text {
// bolt_text_bloom.h
inline constexpr std::uint32_t kBloomWordMaxBytes = 64;
// bolt_text_fn.h
inline constexpr std::uint32_t kFormatMaxWidth = 4096; // width / precision ceiling
// bolt_text_numfmt.h
inline constexpr std::uint32_t kNumFmtMaxItems = 96;       // format items (digits, separators, the point)
// bolt_text_search.h
inline constexpr std::uint32_t kRegexLiteralMax = 64;
inline constexpr std::uint32_t kTokenMaxNeedle = 255;
// bolt_text_spark.h
inline constexpr std::uint32_t kEditMaxChars = 8192;   // levenshtein: code points of the shorter string
}  // namespace bolt::kernels::text

namespace bolt::kernels::variantspark {
// bolt_variant_spark.h
inline constexpr std::uint32_t kSizeLimit = 16u * 1024u * 1024u;   // Spark VARIANT_SIZE_LIMIT, value and metadata
}  // namespace bolt::kernels::variantspark

namespace bolt::kernels::wkb {
// bolt_wkb.h
inline constexpr std::uint32_t k_max_depth = 64u;     // nested geometry collections
inline constexpr std::uint32_t k_max_types = 28u;     // 7 base types x 4 dimension groups
}  // namespace bolt::kernels::wkb

namespace bolt::lakehouse::iceberg {
// metadata.h
inline constexpr std::uint32_t kIcebergSchemaMaxDepth = 32u;
// metrics.h
inline constexpr std::uint32_t kIcebergDefaultMaxInferredColumns = 100u; // write.metadata.metrics.max-inferred-column-defaults
}  // namespace bolt::lakehouse::iceberg

namespace bolt::limits_detail {
// file.cpp
inline constexpr std::size_t kIoOpPool = 256;   // file ops kept for reuse
// bolt_rocm.cpp
inline constexpr std::int32_t kRocmMaxHeadDim = 256;   // two-pass attention
inline constexpr std::int32_t kRocmMaxAttnSeq = 4096;   // scores buffer, 16 KB shared
// bolt_arrow_ipc_reader_internal.h
inline constexpr std::uint32_t kIpcReadDepth = 32;   // list/struct/map nesting
// bolt_parquet_variant.cpp
inline constexpr std::uint32_t kVariantObjectFields = 1u << 16;   // keys of one rebuilt object
// iceberg_metadata_merge.inc
inline constexpr std::uint32_t kIcebergMergeMembers = 128u;   // members of one metadata document, its refs or properties
// bolt_yaml.cpp
inline constexpr std::uint32_t kYamlDepth = 256u;   // nested collections
}  // namespace bolt::limits_detail

namespace bolt::nested_row {
// bolt_nested_row.h
inline constexpr std::uint32_t kMaxCount = 1u << 30;     // elements of one list / map value
inline constexpr std::uint32_t kMaxDepth = 32;
inline constexpr std::uint32_t kMaxNodes = 256;
}  // namespace bolt::nested_row

namespace bolt::net {
// bolt_udp.h
inline constexpr std::uint32_t k_udp_wait_endpoints_max = 64u;
}  // namespace bolt::net

namespace bolt::tz {
// bolt_tz.h
inline constexpr std::int32_t  kMaxFixedOffsetSec = 18 * 3600;   // java.time bound
inline constexpr std::uint32_t kMaxZoneIdChars = 64;
}  // namespace bolt::tz
