// Test-only native consumer of the actual upstream Rust writer's fixture.
#include "bolt/io/bolt_file_io.h"
#include "bolt/wire/xchannel.h"
#include <cassert>
#include <charconv>
#include <cstdio>
#include <cstring>

namespace {
namespace xc = bolt::wire::xchannel;
namespace io = bolt::io;
constexpr std::uint64_t kGeneration = 0x4732584348340001ull;
constexpr std::uint64_t kMaxBytes = 64ull << 20;
constexpr std::uint64_t kMaxRecords = 10000;
constexpr char kName[48] = "bolt-xchannel-v4-fixture";
struct Config {
    const char* base = nullptr;
    std::uint64_t count = 0, payload = 0, region = 0, files = 0;
};
struct State {
    io::MappedFile map{nullptr, 0, -1, -1, 0, 0};
    xc::File file{};
    xc::Cursor cursor{};
    std::uint64_t bytes = 0, records = 0, skips = 0, rolls = 0;
};
std::size_t bounded_length(const char* text, std::size_t limit) noexcept {
    assert(text != nullptr);
    assert(limit <= 960);
    for (std::size_t i = 0; i <= limit; ++i) if (text[i] == '\0') return i;
    return limit + 1;
}
bool number(const char* text, std::uint64_t& out) noexcept {
    assert(text != nullptr);
    const auto size = bounded_length(text, 20);
    if (size == 0 || size > 20) return false;
    const auto result = std::from_chars(text, text + size, out);
    assert(result.ptr >= text && result.ptr <= text + size);
    return result.ec == std::errc{} && result.ptr == text + size;
}
bool arguments(int argc, char** argv, Config& c) noexcept {
    assert(argc >= 1);
    assert(argv != nullptr);
    if (argc != 6) return false;
    c.base = argv[1];
    const auto base_size = bounded_length(c.base, 960);
    if (base_size == 0 || base_size > 960 ||
        !number(argv[2], c.count) || !number(argv[3], c.payload) ||
        !number(argv[4], c.region) || !number(argv[5], c.files)) return false;
    return c.count <= kMaxRecords && c.payload <= 4096 && c.region >= io::io_page_size() &&
        c.region <= kMaxBytes && c.region % io::io_page_size() == 0 &&
        c.files > 0 && c.files <= kMaxRecords + 1;
}
bool open_file(const Config& c, std::uint64_t sequence, std::uint64_t base,
               std::uint64_t remaining, io::MappedFile& map, xc::File& file) noexcept {
    assert(c.base != nullptr);
    assert(remaining <= kMaxBytes);
    char path[1024]{};
    const auto length = sequence == 0 ? std::snprintf(path, sizeof(path), "%s", c.base)
        : std::snprintf(path, sizeof(path), "%s.%llu", c.base,
                        static_cast<unsigned long long>(sequence));
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(path)) return false;
    if (io::mapped_file_open_bounded(path, xc::kPrefixBytes, remaining, &map)
        != io::FileIoStatus::kOk) return false;
    const xc::Limits limits{io::io_page_size(), static_cast<std::uint32_t>(c.region),
                            static_cast<std::uint32_t>(c.payload)};
    const xc::Expected expected{sequence, kGeneration, base, true};
    const auto status = xc::admit_file({map.base, xc::kPrefixBytes}, expected, limits, file);
    if (status != xc::Status::Ok || file.region_size != c.region ||
        std::memcmp(file.name, kName, sizeof(kName)) != 0) {
        io::mapped_file_close(&map);
        return false;
    }
    return true;
}
bool region(const State& s, xc::Region& out) noexcept {
    assert(s.file.region_size != 0);
    assert(s.map.base != nullptr);
    const auto offset = xc::required_region_offset(s.file, s.cursor);
    io::MappedView view{};
    if (io::mapped_view(s.map, offset, s.file.region_size, &view) != io::FileIoStatus::kOk ||
        view.len > UINT32_MAX) return false;
    out = {{view.data, static_cast<std::uint32_t>(view.len)}, offset};
    return true;
}
bool user(const Config& c, State& s, xc::Region view, const xc::Record& r) noexcept {
    assert(c.payload <= 4096);
    assert(s.records <= kMaxRecords);
    const auto index = s.records;
    const auto length = (73 * index + 19) % (c.payload + 1);
    if (index >= c.count || r.record_index != index || r.user_meta_u64 != index ||
        r.message_type != 0x4000 + index % 17 || r.payload.size != length) return false;
    for (std::uint32_t j = 0; j < r.payload.size; ++j) {
        const auto expected = static_cast<std::uint8_t>((index * 17 + j * 31 + (j >> 3)) & 255);
        if (r.payload.data[j] != expected) return false;
    }
    // Repeat a real mapped peek without advancement before accepting ownership.
    xc::Record again{};
    if (xc::peek(s.file, view, s.cursor, again) != xc::Status::Ok ||
        again.offset != r.offset || again.payload.data != r.payload.data ||
        again.record_index != r.record_index) return false;
    if (xc::consume(s.file, view, s.cursor, r) != xc::Status::Ok) return false;
    ++s.records;
    return true;
}
bool roll(const Config& c, State& s, xc::Region view, const xc::Record& r) noexcept {
    assert(s.bytes <= kMaxBytes);
    assert(s.rolls < kMaxRecords + 1);
    if (s.file.sequence + 1 >= c.files) return false;
    io::MappedFile next{nullptr, 0, -1, -1, 0, 0};
    xc::File admitted{};
    if (!open_file(c, r.roll.next_sequence, r.record_index, kMaxBytes - s.bytes,
                   next, admitted)) return false;
    xc::Cursor cursor{};
    if (xc::follow_successor(s.file, view, s.cursor, r, admitted, cursor) != xc::Status::Ok) {
        io::mapped_file_close(&next);
        return false;
    }
    s.bytes += next.size;
    io::mapped_file_close(&s.map);
    s.map = next;
    s.file = admitted;
    s.cursor = cursor;
    ++s.rolls;
    return true;
}
bool verify(const Config& c, State& s) noexcept {
    assert(c.files > 0);
    assert(c.count <= kMaxRecords);
    if (!open_file(c, 0, 0, kMaxBytes, s.map, s.file)) return false;
    s.bytes = s.map.size;
    s.cursor = xc::begin(s.file);
    // Bound all work, including malformed control-only histories.
    constexpr std::uint64_t max_steps = 100000;
    for (std::uint64_t step = 0; step < max_steps; ++step) {
        xc::Region view{};
        if (!region(s, view)) return false;
        xc::Record r{};
        const auto status = xc::peek(s.file, view, s.cursor, r);
        if (status == xc::Status::Idle)
            return s.records == c.count && s.rolls + 1 == c.files &&
                s.file.sequence + 1 == c.files && s.cursor.record_index == c.count;
        if (status != xc::Status::Ok) return false;
        if (r.kind == xc::Kind::User) {
            if (!user(c, s, view, r)) return false;
        } else if (r.kind == xc::Kind::Skip) {
            if (xc::consume(s.file, view, s.cursor, r) != xc::Status::Ok) return false;
            ++s.skips;
        } else if (r.kind == xc::Kind::Roll) {
            if (!roll(c, s, view, r)) return false;
        } else return false;
    }
    return false;
}
} // namespace

int main(int argc, char** argv) {
    Config c{};
    if (!arguments(argc, argv, c)) {
        std::fputs("usage: verify_xchannel_fixture BASE COUNT MAX_PAYLOAD REGION_SIZE EXPECTED_FILES\n",
                   stderr);
        return 2;
    }
    State state{};
    const bool ok = verify(c, state);
    io::mapped_file_close(&state.map);
    std::printf("{\"ok\":%s,\"records\":%llu,\"skips\":%llu,\"rolls\":%llu,"
                "\"mapped_bytes\":%llu,\"format\":4}\n", ok ? "true" : "false",
                static_cast<unsigned long long>(state.records),
                static_cast<unsigned long long>(state.skips),
                static_cast<unsigned long long>(state.rolls),
                static_cast<unsigned long long>(state.bytes));
    return ok ? 0 : 1;
}
