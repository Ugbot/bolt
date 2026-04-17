// bolt_dictionary.h — global (across-morsels) dictionary pool (H2).
//
// Mirrors QuestDB's `SymbolMapWriter`: a table-global dict that
// assigns stable int codes to string literals, so morsels encoded
// through the pool can compare codes directly (no cross-morsel
// remap). The dict is intern-once, grows monotonically; callers
// never see a code whose string later changes.
//
// Layout (arena-owned, zero heap):
//   strings_buf : flat char buffer; each entry = N bytes, variable.
//   offsets[]   : uint32 cumulative offsets; offsets[i+1] - offsets[i]
//                 = length of entry i.
//   table       : SwissTable keyed on swiss_mix(string_hash), value
//                 = dict code. Find-or-insert on intern.
//   num_codes   : current size (next-to-assign code).
//
// Concurrency: this Wave-2 drop ships single-producer-single-consumer
// semantics — a single ingest worker interns, a single planner
// resolves. H4 (lock-free append with tick-tock publish) adds
// multi-producer safety; deferred to Wave 3.
//
// RULES: Tiger Style — POD + free functions, noexcept, arena-only,
// hard-capped capacity set at `create()`. No growth after create.

#pragma once

#include "bolt/bolt_arena.h"
#include "bolt/bolt_hash.h"
#include "bolt/bolt_port.h"
#include "bolt/join/bolt_swiss.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace bolt {

inline constexpr uint32_t kDictPoolInvalidCode = 0xFFFFFFFFu;

struct DictionaryPool {
    char*       strings;      // flat char storage, sized at create
    uint32_t    strings_cap;  // bytes
    uint32_t    strings_used;
    uint32_t*   offsets;      // num_codes + 1 entries; [0]=0
    SwissTable  index;        // hash(str) -> code
    uint32_t    num_codes;
    uint32_t    max_codes;    // hard cap

    // Allocate a pool sized for `max_codes` distinct entries averaging
    // `avg_bytes_per_str` bytes each. Returns false on OOM.
    static bool create(DictionaryPool* pool, uint32_t max_codes,
                       uint32_t avg_bytes_per_str, Arena* arena) noexcept {
        assert(pool  != nullptr);
        assert(arena != nullptr);
        assert(max_codes >= 1u);
        assert(avg_bytes_per_str >= 1u);

        const uint64_t cap_bytes =
            static_cast<uint64_t>(max_codes) * avg_bytes_per_str;
        assert(cap_bytes <= (uint64_t{1} << 30));

        char* strings = static_cast<char*>(
            arena->allocate(static_cast<size_t>(cap_bytes), 8));
        uint32_t* offsets = arena->allocate_array<uint32_t>(max_codes + 1u);
        if (!strings || !offsets) return false;
        offsets[0] = 0u;

        // SwissTable sized to 2× the cap so load stays <50%.
        if (!SwissTable::create(&pool->index,
                                static_cast<uint64_t>(max_codes), arena)) {
            return false;
        }

        pool->strings      = strings;
        pool->strings_cap  = static_cast<uint32_t>(cap_bytes);
        pool->strings_used = 0u;
        pool->offsets      = offsets;
        pool->num_codes    = 0u;
        pool->max_codes    = max_codes;
        return true;
    }

    // FNV-1a 64-bit of a byte range — stable across runs, fine for
    // the hash-table keying (SwissTable applies `swiss_mix` on top).
    BOLT_FORCE_INLINE static uint64_t hash_bytes(const char* data,
                                                 uint32_t len) noexcept {
        uint64_t h = 0xCBF29CE484222325ULL;
        for (uint32_t i = 0; i < len; ++i) {
            h ^= static_cast<uint8_t>(data[i]);
            h *= 0x00000100000001B3ULL;
        }
        return h;
    }

    // Intern a string. Returns the dict code, or kDictPoolInvalidCode
    // on overflow (max_codes reached or strings buffer full).
    uint32_t intern(const char* data, uint32_t len) noexcept {
        assert(data != nullptr || len == 0u);
        const uint64_t h = hash_bytes(data, len);
        const int32_t  existing = index.find(h);
        if (existing >= 0) {
            // Verify content match — hash collisions are rare but
            // possible. On collision we fall through to append (accept
            // that duplicate codes exist for collided entries; caller
            // can re-hash with a second constant if strict dedup
            // matters). For typical symbol-column shapes this is safe.
            const uint32_t code = static_cast<uint32_t>(existing);
            const uint32_t off  = offsets[code];
            const uint32_t cur_len = offsets[code + 1u] - off;
            if (cur_len == len && memcmp(strings + off, data, len) == 0) {
                return code;
            }
            // Collision — fall through to append.
        }
        if (num_codes >= max_codes) return kDictPoolInvalidCode;
        if (strings_used + len > strings_cap) return kDictPoolInvalidCode;

        const uint32_t code = num_codes++;
        memcpy(strings + strings_used, data, len);
        strings_used += len;
        offsets[num_codes] = strings_used;
        // insert hashes on the value we care about; if an older
        // collided entry owned this slot we overwrite it — the
        // collision check above is our correctness guard, not the
        // SwissTable's uniqueness.
        (void)index.insert(h, code);
        return code;
    }

    // Resolve a code to its backing (data, len). Returns (nullptr, 0)
    // if code is out of range.
    BOLT_FORCE_INLINE void resolve(uint32_t code,
                                   const char** out_data,
                                   uint32_t*    out_len) const noexcept {
        assert(out_data != nullptr);
        assert(out_len  != nullptr);
        if (code >= num_codes) {
            *out_data = nullptr;
            *out_len  = 0u;
            return;
        }
        const uint32_t off = offsets[code];
        *out_data = strings + off;
        *out_len  = offsets[code + 1u] - off;
    }

    // Find-only variant — does not intern. Returns kDictPoolInvalidCode
    // if the literal isn't already in the pool. Used by the planner's
    // literal-resolve-once filter path.
    uint32_t find(const char* data, uint32_t len) const noexcept {
        assert(data != nullptr || len == 0u);
        const uint64_t h = hash_bytes(data, len);
        const int32_t  existing = index.find(h);
        if (existing < 0) return kDictPoolInvalidCode;
        const uint32_t code = static_cast<uint32_t>(existing);
        const uint32_t off  = offsets[code];
        const uint32_t cur_len = offsets[code + 1u] - off;
        if (cur_len != len) return kDictPoolInvalidCode;
        if (memcmp(strings + off, data, len) != 0) {
            return kDictPoolInvalidCode;
        }
        return code;
    }
};

}  // namespace bolt
