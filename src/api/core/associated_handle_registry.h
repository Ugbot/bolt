// associated_handle_registry.h — one-time OS-handle association transaction.
//
// This is state logic, not a concurrent container. Callers must serialize every
// ensure()/close_and_forget()/forget()/contains()/size() call with the same
// external lock. Transaction callbacks must not re-enter this registry.

#pragma once

#include "../net/fd_registry.h"

#include <cassert>
#include <cstdint>

namespace bolt::api::core {

enum class AssociationResult : uint8_t {
    already_associated,
    associated,
    registry_refused,
    platform_failed,
};

class AssociatedHandleRegistry {
public:
    static constexpr uint32_t kCapacity =
        net::FdRegistry<uint8_t>::kMaxSegs * net::FdRegistry<uint8_t>::kSegSize;

    template <typename AssociateFn>
    AssociationResult ensure(uint64_t handle, AssociateFn&& associate) noexcept {
        uint8_t* existing = handles_.find(handle);
        if (existing != nullptr) {
            assert(*existing == kAssociatedMarker);
            return AssociationResult::already_associated;
        }

        uint8_t* marker = handles_.insert(handle);
        if (marker == nullptr) return AssociationResult::registry_refused;
        assert(*marker == 0);

        if (!associate()) {
            const bool erased = handles_.erase(handle);
            assert(erased);
            (void)erased;
            return AssociationResult::platform_failed;
        }

        *marker = kAssociatedMarker;
        assert(handles_.find(handle) == marker);
        return AssociationResult::associated;
    }

    // Run the platform close first and remove the association only if it
    // succeeds. This preserves the cache entry when the OS still owns the
    // handle. The caller's external lock must cover this complete transaction.
    template <typename CloseFn>
    bool close_and_forget(uint64_t handle, CloseFn&& close) noexcept {
        const bool was_associated = handles_.find(handle) != nullptr;
        if (!close()) {
            assert((handles_.find(handle) != nullptr) == was_associated);
            return false;
        }

        const bool erased = handles_.erase(handle);
        assert(erased == was_associated);
        assert(handles_.find(handle) == nullptr);
        (void)was_associated;
        (void)erased;
        return true;
    }

    bool forget(uint64_t handle) noexcept { return handles_.erase(handle); }

    bool contains(uint64_t handle) noexcept {
        uint8_t* marker = handles_.find(handle);
        assert(marker == nullptr || *marker == kAssociatedMarker);
        return marker != nullptr;
    }

    uint32_t size() const noexcept { return handles_.size(); }

private:
    static constexpr uint8_t kAssociatedMarker = 1;
    net::FdRegistry<uint8_t> handles_;
};

}  // namespace bolt::api::core
