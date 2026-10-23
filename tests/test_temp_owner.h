#pragma once
#include "bolt/bolt_port.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cassert>
#include <filesystem>
#include <string>
#include <system_error>

namespace bolt::test {
// Cold test-only owner. The caller must fatally check valid() before using its
// path; failure never falls back to /tmp or an unowned working directory.
class TempOwner {
public:
    explicit TempOwner(const char* tag, bool enabled = true) {
        assert(tag != nullptr);
        assert(tag[0] != 0);
        if (!enabled) return;
        const auto base = std::filesystem::temp_directory_path(error_);
        if (error_) return;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0u; attempt < 64u; ++attempt) {
            directory_ = base / (std::string("bolt-") + tag + "-" +
                std::to_string(static_cast<long long>(bolt_getpid())) + "-" +
                std::to_string(tick) + "-" + std::to_string(serial_.fetch_add(1u)));
            error_.clear();
            if (std::filesystem::create_directory(directory_, error_)) {
                valid_ = true;
                path_ = (directory_ / "data").string();
                assert(!path_.empty());
                return;
            }
            if (error_ && error_ != std::errc::file_exists) return;
        }
        error_ = std::make_error_code(std::errc::file_exists);
    }
    ~TempOwner() {
        if (!valid_) return;
        std::error_code cleanup;
        std::filesystem::remove_all(directory_, cleanup);
        EXPECT_FALSE(cleanup) << "temporary owner cleanup: " << cleanup.message();
    }
    TempOwner(const TempOwner&) = delete;
    TempOwner& operator=(const TempOwner&) = delete;
    bool valid() const noexcept { return valid_; }
    const std::string& path() const noexcept { return path_; }
    const std::filesystem::path& directory() const noexcept { return directory_; }
    const std::error_code& error() const noexcept { return error_; }
private:
    inline static std::atomic<uint64_t> serial_{0u};
    std::filesystem::path directory_;
    std::string path_;
    std::error_code error_;
    bool valid_ = false;
};
} // namespace bolt::test
