// bolt/lakehouse/iceberg_delete_file.cpp — position + equality delete loaders.
// W4 scope: data structures + binary-search lookup; Parquet-body loaders TODO(W5).

#include "bolt/lakehouse/iceberg/delete_file.h"

#include <cassert>
#include <cstring>

namespace bolt {
namespace lakehouse {
namespace iceberg {

namespace {

int cmp_path_pos(const PositionDeleteEntry& a, const char* fp,
                 int64_t pos) noexcept {
    const int c = std::strcmp(a.file_path, fp);
    if (c != 0) return c;
    if (a.pos < pos) return -1;
    if (a.pos > pos) return 1;
    return 0;
}

}  // namespace

bool position_delete_set_init(PositionDeleteSet* s, PositionDeleteEntry* buf,
                              uint32_t cap) noexcept {
    assert(s != nullptr && buf != nullptr);
    s->entries = buf; s->n = 0; s->cap = cap;
    return true;
}

bool position_delete_set_contains(const PositionDeleteSet* s,
                                  const char* file_path,
                                  int64_t pos) noexcept {
    assert(s != nullptr && file_path != nullptr);
    if (s->n == 0) return false;
    if (s->n <= 32) {
        for (uint32_t i = 0; i < s->n; ++i) {
            if (cmp_path_pos(s->entries[i], file_path, pos) == 0) return true;
        }
        return false;
    }
    uint32_t lo = 0, hi = s->n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        const int c = cmp_path_pos(s->entries[mid], file_path, pos);
        if (c == 0) return true;
        if (c < 0) lo = mid + 1u; else hi = mid;
    }
    return false;
}

bool equality_delete_set_init(EqualityDeleteSetI64* s, EqualityDeleteI64* buf,
                              uint32_t cap) noexcept {
    assert(s != nullptr && buf != nullptr);
    s->entries = buf; s->n = 0; s->cap = cap;
    return true;
}

bool equality_delete_set_contains(const EqualityDeleteSetI64* s,
                                  int32_t field_id, int64_t value) noexcept {
    assert(s != nullptr);
    for (uint32_t i = 0; i < s->n; ++i) {
        if (s->entries[i].field_id == field_id &&
            s->entries[i].value == value) return true;
    }
    return false;
}

}  // namespace iceberg
}  // namespace lakehouse
}  // namespace bolt
