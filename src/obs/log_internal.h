#pragma once

#include "bolt/obs/log.h"

#if BOLT_OBS_LOG
namespace bolt::log::detail {

inline constexpr std::size_t kFormatBytes = 2048;

void format_message(const Record& r, char* out, std::size_t cap, std::size_t* out_n) noexcept;
std::size_t format_line(const Record& r, Format format, const char* module, const char* service,
                        char* out, std::size_t cap) noexcept;

}  // namespace bolt::log::detail
#endif
