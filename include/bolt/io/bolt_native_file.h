#pragma once
#include <cstdint>

namespace bolt { namespace io {
// Lexical single ownership. Never copy a live token. Adoption consumes only on
// success; access permissions and actual direct mode are supplied by the caller.
struct NativeFile { intptr_t handle = -1; };
static_assert(sizeof(NativeFile) == sizeof(intptr_t));
// Consumes even on OS close error. Invalid is idempotent; null returns false.
bool native_file_close(NativeFile* file) noexcept;
} } // namespace bolt::io
