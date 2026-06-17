// bolt/lakehouse/delta/generated_column.h — recompute generated columns.

#pragma once

#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/lakehouse/delta/log.h"
#include "bolt/lakehouse/format.h"

namespace bolt {
namespace lakehouse {
namespace delta {

enum class GenExprKind : uint8_t {
    kUnknown = 0, kIdentity = 1, kYearOfDate = 2,
    kMonthOfDate = 3, kAddConst = 4,
};

struct GeneratedColumn {
    char         output_name[kLakeMaxColName];
    char         input_name[kLakeMaxColName];
    GenExprKind  kind;
    uint8_t      _pad[7];
    int64_t      const_addend;
};

static constexpr uint32_t kDeltaMaxGenCols = 16u;

struct GeneratedColumnSet {
    GeneratedColumn cols[kDeltaMaxGenCols];
    uint32_t        n_cols;
    uint32_t        _pad;
};

bool delta_generated_cols_build(const DeltaMetadata* md, Arena* scratch,
                                GeneratedColumnSet* out) noexcept;

bool delta_generated_eval_i64(const GeneratedColumn* g, int64_t input,
                              int64_t* out) noexcept;

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
