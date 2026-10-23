// test_bolt_aligned_writer.cpp — MSEG K9: page CRC32C (3-way interleaved,
// zero padding included) and the aligned direct-I/O page writer.
//
// Throughput is reported, not gated, unless BOLT_PERF_GATE=1 (the CRC floor
// is then 11.5 GB/s on one thread). BOLT_AW_BENCH_MB sizes the write bench.

#include "bolt/io/bolt_aligned_writer.h"
#include "bolt/io/bolt_crc32c.h"
#include "test_temp_owner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using bolt::io::AlignedPage;
using bolt::io::AlignedWriteStatus;
using bolt::io::AlignedWriter;
using bolt::io::AlignedWriterConfig;
using bolt::io::AlignUnit;

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
    out->clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    const int end = std::fseek(f, 0, SEEK_END);
    const long n = end == 0 ? std::ftell(f) : -1;
    const int begin = n >= 0 ? std::fseek(f, 0, SEEK_SET) : -1;
    if (n < 0 || begin != 0) { std::fclose(f); return false; }
    out->resize(static_cast<size_t>(n));
    const size_t got = n > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
    const int closed = std::fclose(f);
    return got == out->size() && closed == 0;
}

void fill_random(std::vector<uint8_t>& v, std::mt19937_64& rng) {
    for (auto& b : v) b = static_cast<uint8_t>(rng());
}

// ---------------------------------------------------------------------------
// CRC
// ---------------------------------------------------------------------------

TEST(PageCrc, ThreeWayMatchesSoftwareOnEveryLengthAndOffset) {
    std::mt19937_64 rng(0x5EED9u);
    std::vector<uint8_t> buf(64 * 1024 + 64);
    fill_random(buf, rng);
    const size_t L = bolt::io::kCrc32cLaneBytes;
    std::vector<size_t> lens = {0, 1, 7, 8, 63, L, 3 * L - 1, 3 * L, 3 * L + 1, 3 * L + 7,
                                6 * L, 6 * L + 13, 16384, 65536};
    for (int i = 0; i < 400; ++i) lens.push_back(static_cast<size_t>(rng() % 65536));
    for (size_t len : lens) {
        for (size_t off = 0; off < 8; ++off) {
            const uint32_t seed = static_cast<uint32_t>(rng());
            const uint32_t want = bolt::io::crc32c_software(buf.data() + off, len, seed);
            ASSERT_EQ(bolt::io::crc32c(buf.data() + off, len, seed), want)
                << "len " << len << " off " << off;
#if BOLT_CRC32C_HW_X86 || BOLT_CRC32C_HW_ARM
            ASSERT_EQ(bolt::io::crc32c_3way(buf.data() + off, len, seed), want)
                << "len " << len << " off " << off;
#endif
        }
    }
    const uint8_t s[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    EXPECT_EQ(bolt::io::crc32c(s, 9), 0xE3069283u);
}

TEST(PageCrc, ExtendZerosEqualsCrcOverZeroBytes) {
    std::vector<uint8_t> zeros(200000, 0);
    std::mt19937_64 rng(7);
    const uint64_t ns[] = {0, 1, 3, 8, 63, 64, 4095, 16384, 65535, 65536, 199999};
    for (uint64_t n : ns) {
        const uint32_t c = static_cast<uint32_t>(rng());
        EXPECT_EQ(bolt::io::crc32c_extend_zeros(c, n),
                  bolt::io::crc32c_software(zeros.data(), n, c))
            << n;
    }
}

TEST(PageCrc, PageCrcCoversZeroPadding) {
    std::mt19937_64 rng(11);
    std::vector<uint8_t> page(20000);
    fill_random(page, rng);
    std::vector<uint8_t> padded(32768, 0);
    std::memcpy(padded.data(), page.data(), page.size());
    EXPECT_EQ(bolt::io::page_crc32c(page.data(), page.size(), padded.size()),
              bolt::io::crc32c(padded.data(), padded.size()));
    EXPECT_NE(bolt::io::page_crc32c(page.data(), page.size(), padded.size()),
              bolt::io::crc32c(page.data(), page.size()));
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

AlignedWriterConfig small_cfg(bool direct) {
    AlignedWriterConfig c = bolt::io::aligned_writer_config_default();
    c.direct_io = direct ? 1 : 0;
    c.staging_bytes = 4u * (1u << c.stripe_log2);  // flushes mid-test
    return c;
}

struct Written {
    AlignedPage page;
    uint32_t unit;
};

// Write a random sequence of pages, raw appends and pads; return the pages.
bool write_random_file(const std::string& path, const AlignedWriterConfig& cfg,
                       uint64_t* size, uint8_t* direct, uint64_t seed,
                       std::vector<Written>* pages) {
    *size = 0; *direct = 0; pages->clear();
    std::mt19937_64 rng(seed);
    AlignedWriter w{};
    const auto opened = bolt::io::aligned_writer_open(path.c_str(), cfg, &w);
    if (opened != AlignedWriteStatus::kOk) return false;
    *direct = w.direct;
    std::vector<uint8_t> buf(200000);
    for (int i = 0; i < 300; ++i) {
        const uint32_t kind = static_cast<uint32_t>(rng() % 8);
        const size_t len = static_cast<size_t>(rng() % (kind == 0 ? 150000 : 9000));
        buf.resize(len);
        fill_random(buf, rng);
        if (kind == 7) {
            const auto status = bolt::io::aligned_writer_append(&w, buf.data(), len % 100);
            EXPECT_EQ(status, AlignedWriteStatus::kOk);
            if (status != AlignedWriteStatus::kOk) { bolt::io::aligned_writer_abort(&w); return false; }
            continue;
        }
        const AlignUnit u = static_cast<AlignUnit>(rng() % 3);
        Written wr{};
        wr.unit = w.unit[static_cast<uint32_t>(u)];
        const auto status = bolt::io::aligned_writer_page(&w, buf.data(), len, u, 0, &wr.page);
        EXPECT_EQ(status, AlignedWriteStatus::kOk);
        if (status != AlignedWriteStatus::kOk) { bolt::io::aligned_writer_abort(&w); return false; }
        pages->push_back(wr);
    }
    const auto finished = bolt::io::aligned_writer_finish(&w, true, size);
    EXPECT_EQ(finished, AlignedWriteStatus::kOk);
    if (finished != AlignedWriteStatus::kOk) bolt::io::aligned_writer_abort(&w);
    return finished == AlignedWriteStatus::kOk;
}

void check_file(bool direct_wanted) {
    bolt::test::TempOwner temp(direct_wanted ? "layout_direct" : "layout_buffered");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    uint64_t size = 0;
    uint8_t direct = 0;
    const AlignedWriterConfig cfg = small_cfg(direct_wanted);
    std::vector<Written> pages;
    ASSERT_TRUE(write_random_file(path, cfg, &size, &direct, 42, &pages));
    std::printf("[aligned_writer] direct requested=%d granted=%d\n", direct_wanted, direct);
    if (!direct_wanted) EXPECT_EQ(direct, 0);
    std::vector<uint8_t> file;
    ASSERT_TRUE(read_file(path, &file));
    ASSERT_EQ(file.size(), size);
    std::mt19937_64 rng(42);  // replay the payloads
    std::vector<uint8_t> buf(200000);
    size_t pi = 0;
    for (int i = 0; i < 300; ++i) {
        const uint32_t kind = static_cast<uint32_t>(rng() % 8);
        const size_t len = static_cast<size_t>(rng() % (kind == 0 ? 150000 : 9000));
        buf.resize(len);
        fill_random(buf, rng);
        if (kind == 7) continue;
        (void)rng();  // preserve the unit draw while replaying payloads
        const Written& wr = pages[pi++];
        const AlignedPage& p = wr.page;
        ASSERT_EQ(p.off % wr.unit, 0u);
        ASSERT_EQ(p.padded_len % wr.unit, 0u);
        ASSERT_EQ(p.len, len);
        ASSERT_LE(p.off + p.padded_len, file.size());
        ASSERT_EQ(std::memcmp(file.data() + p.off, buf.data(), len), 0);
        for (uint64_t b = p.off + len; b < p.off + p.padded_len; ++b) ASSERT_EQ(file[b], 0u);
        ASSERT_EQ(bolt::io::crc32c(file.data() + p.off, p.padded_len), p.crc32c);
    }
    ASSERT_EQ(pi, pages.size());
    std::remove(path.c_str());
}

TEST(AlignedWriter, MissingDirectoryGenerationRefusesWithoutPublishingPages) {
    bolt::test::TempOwner temp("aw-missing");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    uint64_t size = 77u; uint8_t direct = 1u;
    std::vector<Written> pages;
    const auto missing = (temp.directory() / "missing" / "data").string();
    EXPECT_FALSE(write_random_file(missing, small_cfg(true), &size, &direct, 99u, &pages));
    EXPECT_TRUE(pages.empty()); EXPECT_EQ(size, 0u); EXPECT_EQ(direct, 0u);
}

TEST(AlignedWriter, OffsetsOnUnitsPaddingZeroBuffered) {
    ASSERT_NO_FATAL_FAILURE(check_file(false));
}
TEST(AlignedWriter, OffsetsOnUnitsPaddingZeroDirect) {
    ASSERT_NO_FATAL_FAILURE(check_file(true));
}

TEST(AlignedWriter, RandomByteFlipsAllDetectedByPageCrc) {
    bolt::test::TempOwner temp("flips");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    uint64_t size = 0;
    uint8_t direct = 0;
    std::vector<Written> pages;
    ASSERT_TRUE(write_random_file(path, small_cfg(true), &size, &direct, 99, &pages));
    std::vector<uint8_t> file;
    ASSERT_TRUE(read_file(path, &file));
    ASSERT_EQ(file.size(), size);
    std::mt19937_64 rng(1234);
    std::vector<size_t> nonempty;
    for (size_t i = 0; i < pages.size(); ++i) if (pages[i].page.padded_len != 0u) nonempty.push_back(i);
    ASSERT_FALSE(nonempty.empty());
    uint32_t detected = 0, flips = 0;
    for (; flips < 10000; ++flips) {
        const AlignedPage& p = pages[nonempty[rng() % nonempty.size()]].page;
        const uint64_t at = p.off + rng() % p.padded_len;  // payload or padding
        const uint8_t x = static_cast<uint8_t>(1 + rng() % 255);
        file[at] ^= x;
        if (bolt::io::crc32c(file.data() + p.off, p.padded_len) != p.crc32c) ++detected;
        file[at] ^= x;
    }
    std::printf("[aligned_writer] byte flips detected %u / %u\n", detected, flips);
    EXPECT_EQ(flips, 10000u);
    EXPECT_EQ(detected, flips);
    std::remove(path.c_str());
}

TEST(AlignedWriter, RefusesBadUnitsAndStaging) {
    bolt::test::TempOwner temp("bad");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    AlignedWriter w{};
    AlignedWriterConfig c = bolt::io::aligned_writer_config_default();
    c.buf_log2 = 5;  // 32 B < 64 B
    EXPECT_EQ(bolt::io::aligned_writer_open(path.c_str(), c, &w), AlignedWriteStatus::kBadUnits);
    c = bolt::io::aligned_writer_config_default();
    c.chunk_log2 = static_cast<uint8_t>(c.stripe_log2 + 1);  // chunk > stripe
    EXPECT_EQ(bolt::io::aligned_writer_open(path.c_str(), c, &w), AlignedWriteStatus::kBadUnits);
    c = bolt::io::aligned_writer_config_default();
    c.staging_bytes = (1u << c.stripe_log2) + 4096u;  // not a stripe multiple
    EXPECT_EQ(bolt::io::aligned_writer_open(path.c_str(), c, &w),
              AlignedWriteStatus::kBadStaging);
    c = bolt::io::aligned_writer_config_default();
    EXPECT_EQ(bolt::io::aligned_writer_open("/nonexistent-dir/x/y", c, &w),
              AlignedWriteStatus::kOpenFailed);
}

TEST(AlignedWriter, EmptyFileAndTailShorterThanIoUnit) {
    bolt::test::TempOwner temp("tail");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    AlignedWriter w{};
    ASSERT_EQ(bolt::io::aligned_writer_open(path.c_str(), small_cfg(true), &w),
              AlignedWriteStatus::kOk);
    uint64_t size = 1;
    ASSERT_EQ(bolt::io::aligned_writer_finish(&w, false, &size), AlignedWriteStatus::kOk);
    EXPECT_EQ(size, 0u);
    std::vector<uint8_t> empty;
    ASSERT_TRUE(read_file(path, &empty));
    EXPECT_EQ(empty.size(), 0u);
    ASSERT_EQ(bolt::io::aligned_writer_open(path.c_str(), small_cfg(true), &w),
              AlignedWriteStatus::kOk);
    const char msg[] = "abc";
    ASSERT_EQ(bolt::io::aligned_writer_append(&w, msg, 3), AlignedWriteStatus::kOk);
    ASSERT_EQ(bolt::io::aligned_writer_finish(&w, false, &size), AlignedWriteStatus::kOk);
    EXPECT_EQ(size, 3u);
    std::vector<uint8_t> f;
    ASSERT_TRUE(read_file(path, &f));
    ASSERT_EQ(f.size(), 3u);
    EXPECT_EQ(std::memcmp(f.data(), msg, 3), 0);
    EXPECT_EQ(bolt::io::aligned_writer_append(&w, msg, 3), AlignedWriteStatus::kClosed);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Throughput (reported; gated only under BOLT_PERF_GATE=1)
// ---------------------------------------------------------------------------

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double crc_gbps(size_t page_bytes) {
    std::vector<uint8_t> buf(page_bytes);
    std::mt19937_64 rng(3);
    fill_random(buf, rng);
    const size_t iters = (size_t(2) << 30) / page_bytes;  // 2 GiB per run
    std::vector<double> runs;
    uint32_t c = 0;
    for (int r = 0; r < 7; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) c = bolt::io::crc32c(buf.data(), page_bytes, c);
        const double s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        runs.push_back(static_cast<double>(page_bytes * iters) / s / 1e9);
    }
    EXPECT_NE(c, 0x12345678u);  // keep the loop live
    return median(runs);
}

TEST(AlignedWriterPerf, PageCrcThroughputOneThread) {
    const double g16 = crc_gbps(16384), g64 = crc_gbps(65536), g1m = crc_gbps(1 << 20);
    std::printf("[aligned_writer] page CRC32C 1 thread median of 7: 16 KiB %.2f GB/s, "
                "64 KiB %.2f GB/s, 1 MiB %.2f GB/s\n", g16, g64, g1m);
    const char* gate = std::getenv("BOLT_PERF_GATE");
    if (gate != nullptr && std::strcmp(gate, "1") == 0) EXPECT_GE(g64, 11.5);
}

bool write_gbps(bool direct, size_t total, uint8_t* granted, double* gbps) {
    *granted = 0; *gbps = 0;
    bolt::test::TempOwner temp(direct ? "bench_direct" : "bench_buffered");
    if (!temp.valid()) return false;
    const std::string& path = temp.path();
    std::vector<uint8_t> page(65536);
    std::mt19937_64 rng(5);
    fill_random(page, rng);
    AlignedWriterConfig c = bolt::io::aligned_writer_config_default();
    c.direct_io = direct ? 1 : 0;
    AlignedWriter w{};
    const auto t0 = std::chrono::steady_clock::now();
    const auto opened = bolt::io::aligned_writer_open(path.c_str(), c, &w);
    EXPECT_EQ(opened, AlignedWriteStatus::kOk);
    if (opened != AlignedWriteStatus::kOk) return false;
    *granted = w.direct;
    AlignedPage page_out{};
    for (size_t done = 0; done < total; done += page.size()) {
        const auto status = bolt::io::aligned_writer_page(&w, page.data(), page.size(), AlignUnit::kStripe, 0, &page_out);
        EXPECT_EQ(status, AlignedWriteStatus::kOk);
        if (status != AlignedWriteStatus::kOk) { bolt::io::aligned_writer_abort(&w); return false; }
    }
    const auto finished = bolt::io::aligned_writer_finish(&w, true, nullptr);
    EXPECT_EQ(finished, AlignedWriteStatus::kOk);
    if (finished != AlignedWriteStatus::kOk) { bolt::io::aligned_writer_abort(&w); return false; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    *gbps = static_cast<double>(total) / seconds / 1e9;
    return true;
}

TEST(AlignedWriterPerf, DirectAndBufferedWriteThroughput) {
    const char* mb = std::getenv("BOLT_AW_BENCH_MB");
    const size_t total = static_cast<size_t>(mb != nullptr ? std::atoi(mb) : 128) << 20;
    std::vector<double> d, b;
    uint8_t gd = 0, gb = 0;
    for (int r = 0; r < 5; ++r) {
        double direct_gbps = 0, buffered_gbps = 0;
        ASSERT_TRUE(write_gbps(true, total, &gd, &direct_gbps));
        ASSERT_TRUE(write_gbps(false, total, &gb, &buffered_gbps));
        d.push_back(direct_gbps);
        b.push_back(buffered_gbps);
    }
    std::printf("[aligned_writer] write %zu MiB + fsync, median of 5: direct(granted=%d) "
                "%.2f GB/s, buffered %.2f GB/s\n", total >> 20, gd, median(d), median(b));
    EXPECT_EQ(gb, 0);
}


bolt::io::NativeFile native_write_open(const char* path) {
#if defined(_WIN32)
    return bolt::io::NativeFile{reinterpret_cast<intptr_t>(CreateFileA(path,
        GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr))};
#else
    return bolt::io::NativeFile{::open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600)};
#endif
}

TEST(AlignedWriter, NativeAdoptionFailurePreservesAndFinishTransfers) {
    using namespace bolt::io;
    bolt::test::TempOwner temp("native_adoption");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    NativeFile n = native_write_open(path.c_str());
    ASSERT_NE(n.handle, -1);
    const intptr_t identity = n.handle;
    AlignedWriter w{};
    auto cfg = aligned_writer_config_default();
    auto bad = cfg;
    bad.staging_bytes = 1;
    EXPECT_EQ(aligned_writer_adopt(&n, bad, false, &w), AlignedWriteStatus::kBadStaging);
    EXPECT_EQ(n.handle, identity);
    EXPECT_EQ(w.handle, -1);
    bad = cfg;
    bad.io_log2 = 0;
    EXPECT_EQ(aligned_writer_adopt(&n, bad, false, &w), AlignedWriteStatus::kBadUnits);
    EXPECT_EQ(aligned_writer_adopt(nullptr, cfg, false, &w), AlignedWriteStatus::kOpenFailed);
    EXPECT_EQ(aligned_writer_adopt(&n, cfg, false, nullptr), AlignedWriteStatus::kOpenFailed);
    ASSERT_EQ(aligned_writer_adopt(&n, cfg, false, &w), AlignedWriteStatus::kOk);
    EXPECT_EQ(w.handle, identity);
    EXPECT_EQ(w.direct, 0);
    EXPECT_EQ(n.handle, -1);
    EXPECT_EQ(aligned_writer_adopt(reinterpret_cast<NativeFile*>(&w), cfg, false, &w),
              AlignedWriteStatus::kOpenFailed);
    EXPECT_EQ(aligned_writer_adopt(&n, cfg, false, &w), AlignedWriteStatus::kOpenFailed);
    const uint8_t data[] = {9, 8, 7};
    ASSERT_EQ(aligned_writer_append(&w, data, sizeof(data)), AlignedWriteStatus::kOk);
    uint64_t size = 0;
    EXPECT_EQ(aligned_writer_finish(&w, true, &size), AlignedWriteStatus::kOk);
    EXPECT_EQ(size, sizeof(data));
    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_file(path, &actual));
    EXPECT_EQ(actual, std::vector<uint8_t>({9, 8, 7}));
    aligned_writer_abort(&w);
    EXPECT_TRUE(native_file_close(&n));
    std::remove(path.c_str());
}

TEST(AlignedWriter, NativeNonemptyRefusesWithoutTruncationAndAbortCloses) {
    using namespace bolt::io;
    bolt::test::TempOwner temp("native_nonempty");
    ASSERT_TRUE(temp.valid()) << temp.error().message();
    const std::string& path = temp.path();
    NativeFile n = native_write_open(path.c_str());
    ASSERT_NE(n.handle, -1);
    const uint8_t data[] = {5, 4, 3, 2};
#if defined(_WIN32)
    DWORD written = 0;
    ASSERT_TRUE(WriteFile(reinterpret_cast<HANDLE>(n.handle), data, sizeof(data), &written, nullptr));
    ASSERT_EQ(written, sizeof(data));
#else
    ASSERT_EQ(::write(static_cast<int>(n.handle), data, sizeof(data)), sizeof(data));
#endif
    const intptr_t identity = n.handle;
    AlignedWriter w{};
    const auto cfg = aligned_writer_config_default();
    EXPECT_EQ(aligned_writer_adopt(&n, cfg, false, &w), AlignedWriteStatus::kOpenFailed);
    EXPECT_EQ(n.handle, identity);
    EXPECT_TRUE(native_file_close(&n));
    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_file(path, &actual));
    EXPECT_EQ(actual, std::vector<uint8_t>({5, 4, 3, 2}));
    n = native_write_open(path.c_str());
    ASSERT_EQ(aligned_writer_adopt(&n, cfg, false, &w), AlignedWriteStatus::kOk);
    ASSERT_EQ(aligned_writer_append(&w, data, sizeof(data)), AlignedWriteStatus::kOk);
    aligned_writer_abort(&w);
    aligned_writer_abort(&w);
    ASSERT_TRUE(read_file(path, &actual));
    EXPECT_TRUE(actual.empty());
#if !defined(_WIN32)
    NativeFile dir{::open("/tmp", O_RDONLY | O_DIRECTORY)};
    ASSERT_NE(dir.handle, -1);
    EXPECT_EQ(aligned_writer_adopt(&dir, cfg, false, &w), AlignedWriteStatus::kOpenFailed);
    EXPECT_TRUE(native_file_close(&dir));
#endif
    std::remove(path.c_str());
}

}  // namespace
