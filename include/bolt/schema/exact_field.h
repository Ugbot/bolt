#pragma once
#include <cstdint>
namespace bolt::schema {
// Metadata only: no row inference, planner identity or allocation. Enum values
// are codec ABI; unknown values refuse rather than falling back to strings.
enum class Physical : uint8_t { Bool=1, Signed, Unsigned, Float, Decimal, Utf8, Binary, Temporal, Group };
enum class Temporal : uint8_t { None, Timestamp, Duration, Date, Time, TimestampUtc };
enum class Unit : uint8_t { None, Second, Millisecond, Microsecond, Nanosecond, Day };
enum class Presence : uint8_t { Required, Nullable, Optional };
struct ExactField {
    Physical physical = Physical::Signed;
    uint16_t bits = 64;
    uint16_t precision = 0;
    int16_t scale = 0;
    Temporal temporal = Temporal::None;
    Unit unit = Unit::None;
    Presence presence = Presence::Required;
    uint8_t sentinel_bytes = 0;
    uint8_t sentinel[32]{}; // exact raw representation, never a JSON number
    uint32_t fixed_length = 0;
    uint16_t parent_plus_one = 0; // 0 = root; otherwise earlier descriptor index+1
    uint32_t group_max_count = 0;
    uint32_t since_version = 0;
    // Interpretation references resolve immutable decoder-owned manifests.
    uint64_t encoding_domain = 0;
    uint64_t epoch_calendar = 0;
    uint64_t timezone = 0;
};
} // namespace bolt::schema
