// test_bolt_file_io.cpp — MSEG B6: positional + direct reads, mmap views and
// readahead hints, range_coalesce properties, batched reads == serial reads.
//
// Throughput is reported, not gated, unless BOLT_PERF_GATE=1 (cold 1 MiB
// aligned sequential pread then >= 5.1 GB/s, layout decision §3.13).
// BOLT_FIO_BENCH_MB sizes the bench file (default 256; BOLT_FIO_BENCH_FILE
// reads an existing file of at least that size instead), BOLT_FIO_THREADS the
// wide run (default 12), BOLT_FIO_FUZZ_ITERS the batch fuzz.

#include "bolt/io/bolt_aligned_writer.h"
#include "bolt/io/bolt_file_io.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

using namespace bolt::io;

uint64_t env_u64(const char* name, uint64_t dflt) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::strtoull(v, nullptr, 10) : dflt;
}

std::string tmp_path(const char* tag) {
    const char* d = std::getenv("TMPDIR");
    std::string p = (d != nullptr && *d != '\0') ? d : "/tmp";
    if (p.back() != '/') p += '/';
#if defined(_WIN32)
    p += std::string("bolt_fio_") + tag;
#else
    p += std::string("bolt_fio_") + tag + "_" + std::to_string(::getpid());
#endif
    return p;
}

std::vector<uint8_t> random_bytes(size_t n, uint64_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= n; i += 8) {
        const uint64_t x = rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    for (size_t i = n & ~size_t(7); i < n; ++i) v[i] = static_cast<uint8_t>(rng());
    return v;
}

void write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(std::fwrite(bytes.data(), 1, bytes.size(), f), bytes.size());
    std::fclose(f);
}

struct AlignedBuf {
    uint8_t* p = nullptr;
    AlignedBuf(size_t n, size_t a) : p(io_alloc_aligned(n, a)) {}
    ~AlignedBuf() { io_free_aligned(p); }
};

// --- positional reads ---------------------------------------------------------

TEST(FileIo, PreadMatchesFileBufferedAndDirect) {
    const std::string path = tmp_path("pread");
    const size_t n = (3u << 20) + 12345;
    const std::vector<uint8_t> bytes = random_bytes(n, 1);
    write_file(path, bytes);
    for (int direct = 0; direct < 2; ++direct) {
        ReadFile f{};
        ASSERT_EQ(file_open_read(path.c_str(), direct != 0, &f), FileIoStatus::kOk);
        EXPECT_EQ(f.size, n);
        EXPECT_EQ(f.io_unit, bolt::wire::kFrameIoAlign);
        AlignedBuf buf(1u << 20, f.io_unit);
        ASSERT_NE(buf.p, nullptr);
        std::mt19937_64 rng(7u + static_cast<unsigned>(direct));
        for (int i = 0; i < 200; ++i) {
            const uint64_t unit = f.io_unit;
            const uint64_t off = (rng() % (n / unit + 1)) * unit;
            const uint64_t len = ((rng() % 64) + 1) * unit;
            uint64_t got = 0;
            ASSERT_EQ(file_pread_aligned(f, off, buf.p, len, &got), FileIoStatus::kOk);
            const uint64_t want = std::min<uint64_t>(len, n > off ? n - off : 0);
            ASSERT_EQ(got, want) << "off " << off << " len " << len;
            ASSERT_EQ(std::memcmp(buf.p, bytes.data() + off, got), 0);
        }
        file_close(&f);
        file_close(&f);
        EXPECT_EQ(f.handle, -1);
    }
    std::remove(path.c_str());
}

TEST(FileIo, PreadUnalignedBufferedAndShortAtEof) {
    const std::string path = tmp_path("unaligned");
    const std::vector<uint8_t> bytes = random_bytes(100003, 2);
    write_file(path, bytes);
    ReadFile f{};
    ASSERT_EQ(file_open_read(path.c_str(), false, &f), FileIoStatus::kOk);
    std::vector<uint8_t> dst(5000);
    uint64_t got = 0;
    ASSERT_EQ(file_pread(f, 777, dst.data() + 3, 4000, &got), FileIoStatus::kOk);
    EXPECT_EQ(got, 4000u);
    EXPECT_EQ(std::memcmp(dst.data() + 3, bytes.data() + 777, 4000), 0);
    ASSERT_EQ(file_pread(f, 99000, dst.data(), 5000, &got), FileIoStatus::kOk);
    EXPECT_EQ(got, 1003u);
    EXPECT_EQ(std::memcmp(dst.data(), bytes.data() + 99000, 1003), 0);
    ASSERT_EQ(file_pread(f, 200000, dst.data(), 10, &got), FileIoStatus::kOk);
    EXPECT_EQ(got, 0u);
    ASSERT_EQ(file_pread(f, 0, dst.data(), 0, &got), FileIoStatus::kOk);
    EXPECT_EQ(file_advise_willneed(f, 0, bytes.size()), FileIoStatus::kOk);
    file_close(&f);
    std::remove(path.c_str());
}

TEST(FileIo, AlignedReadRejectsMisalignment) {
    const std::string path = tmp_path("misalign");
    write_file(path, random_bytes(1u << 18, 3));
    ReadFile f{};
    ASSERT_EQ(file_open_read(path.c_str(), true, &f), FileIoStatus::kOk);
    AlignedBuf buf(1u << 17, f.io_unit);
    uint64_t got = 0;
    EXPECT_EQ(file_pread_aligned(f, 64, buf.p, f.io_unit, &got), FileIoStatus::kMisaligned);
    EXPECT_EQ(file_pread_aligned(f, 0, buf.p + 64, f.io_unit, &got), FileIoStatus::kMisaligned);
    EXPECT_EQ(file_pread_aligned(f, 0, buf.p, f.io_unit + 1, &got), FileIoStatus::kMisaligned);
    EXPECT_EQ(file_pread_aligned(f, 0, buf.p, f.io_unit, &got), FileIoStatus::kOk);
    EXPECT_EQ(got, f.io_unit);
    file_close(&f);
    EXPECT_EQ(file_open_read((path + ".missing").c_str(), false, &f), FileIoStatus::kOpenFailed);
    std::remove(path.c_str());
}

// --- mmap --------------------------------------------------------------------------

TEST(FileIo, MappedViewsArePageAligned) {
    const std::string path = tmp_path("mmap");
    const size_t n = (1u << 20) + 999;
    const std::vector<uint8_t> bytes = random_bytes(n, 4);
    write_file(path, bytes);
    MappedFile m{};
    ASSERT_EQ(mapped_file_open(path.c_str(), &m), FileIoStatus::kOk);
    ASSERT_EQ(m.size, n);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(m.base) % io_page_size(), 0u);
    std::mt19937_64 rng(9);
    for (int i = 0; i < 500; ++i) {
        const uint64_t off = rng() % n;
        const uint64_t len = rng() % (200000);
        MappedView v{};
        ASSERT_EQ(mapped_view(m, off, len, &v), FileIoStatus::kOk);
        ASSERT_EQ(v.len, std::min<uint64_t>(len, n - off));
        ASSERT_EQ(v.data, m.base + off);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(v.page_base) % m.page, 0u);
        ASSERT_EQ(v.page_len % m.page, 0u);
        ASSERT_LE(v.page_base, v.data);
        ASSERT_GE(v.page_base + v.page_len, v.data + v.len);
        ASSERT_LT(v.data - v.page_base, static_cast<ptrdiff_t>(m.page));
        ASSERT_EQ(std::memcmp(v.data, bytes.data() + off, v.len), 0);
        ASSERT_EQ(mapped_advise(m, off, len, static_cast<Advice>(rng() % 4)), FileIoStatus::kOk);
    }
    MappedView v{};
    EXPECT_EQ(mapped_view(m, n + 1, 1, &v), FileIoStatus::kBadArg);
    mapped_file_close(&m);
    mapped_file_close(&m);
    EXPECT_EQ(m.base, nullptr);
    std::remove(path.c_str());
}

#if !defined(_WIN32)
TEST(FileIo, EvictDropsResidency) {
    const std::string path = tmp_path("evict");
    const size_t n = 8u << 20;
    write_file(path, random_bytes(n, 5));
    MappedFile m{};
    ASSERT_EQ(mapped_file_open(path.c_str(), &m), FileIoStatus::kOk);
    uint64_t sum = 0;
    for (size_t i = 0; i < n; i += m.page) sum += m.base[i];
    uint64_t res = 0, pages = 0;
    ASSERT_EQ(mapped_resident_pages(m, 0, n, &res, &pages), FileIoStatus::kOk);
    EXPECT_EQ(pages, (n + m.page - 1) / m.page);
    EXPECT_EQ(res, pages) << sum;
    mapped_file_close(&m);
    ASSERT_EQ(file_evict_cache(path.c_str()), FileIoStatus::kOk);
    ASSERT_EQ(mapped_file_open(path.c_str(), &m), FileIoStatus::kOk);
    ASSERT_EQ(mapped_resident_pages(m, 0, n, &res, &pages), FileIoStatus::kOk);
    EXPECT_LE(res * 10, pages) << "resident " << res << "/" << pages;
    mapped_file_close(&m);
    std::remove(path.c_str());
}
#endif

// --- range_coalesce ----------------------------------------------------------------

// Every request byte lies in an input or in a gap <= max_gap between inputs.
void check_coalesce(const std::vector<IoRange>& in, uint64_t max_req, uint64_t max_gap,
                    const std::vector<CoalescedRange>& out) {
    uint64_t space = 0;
    for (const IoRange& r : in) space = std::max(space, r.off + r.len);
    std::vector<uint8_t> want(space, 0), have(space, 0);
    for (const IoRange& r : in)
        std::fill(want.data() + r.off, want.data() + r.off + r.len, uint8_t(1));
    for (size_t k = 0; k < out.size(); ++k) {
        const CoalescedRange& c = out[k];
        ASSERT_GT(c.len, 0u);
        ASSERT_LE(c.len, max_req);
        ASSERT_LE(c.off + c.len, space);
        if (k > 0) ASSERT_GE(c.off, out[k - 1].off + out[k - 1].len) << "sorted, disjoint";
        ASSERT_GE(c.count, 1u);
        ASSERT_LE(c.first + c.count, in.size());
        for (uint64_t b = c.off; b < c.off + c.len; ++b) have[b] = 1;
        ASSERT_EQ(want[c.off], 1) << "request starts on input bytes";
        ASSERT_EQ(want[c.off + c.len - 1], 1) << "request ends on input bytes";
        uint64_t run = 0;
        for (uint64_t b = c.off; b < c.off + c.len; ++b) {
            run = want[b] ? 0 : run + 1;
            ASSERT_LE(run, max_gap) << "request reads a gap wider than max_gap";
        }
        if (k > 0) {
            const CoalescedRange& p = out[k - 1];
            const uint64_t gap = c.off - (p.off + p.len);
            // A mergeable gap was left only because the merge would pass max_req.
            if (gap <= max_gap) {
                uint64_t end = c.off + c.len;
                const IoRange& lead = in[c.first];
                end = std::max(end, std::min(lead.off + lead.len, p.off + max_req + 1));
                ASSERT_GT(end - p.off, max_req) << "mergeable neighbours not merged, k " << k;
            }
        }
    }
    for (uint64_t b = 0; b < space; ++b) ASSERT_TRUE(!want[b] || have[b]) << "byte " << b;
}

std::vector<CoalescedRange> coalesce(const std::vector<IoRange>& in, uint64_t max_req,
                                     uint64_t max_gap) {
    const uint32_t n = static_cast<uint32_t>(in.size());
    const uint32_t need = range_coalesce_count(in.data(), n, max_req, max_gap);
    std::vector<CoalescedRange> out(need + 1);
    uint32_t got = 0;
    EXPECT_EQ(range_coalesce(in.data(), n, max_req, max_gap, out.data(),
                             static_cast<uint32_t>(out.size()), &got),
              FileIoStatus::kOk);
    EXPECT_EQ(got, need);
    out.resize(got);
    return out;
}

TEST(RangeCoalesce, FixedCases) {
    const uint64_t M = kCoalesceMaxRequestDefault, G = kCoalesceMaxGapDefault;
    auto o = coalesce({{0, 100}, {100 + G, 100}}, M, G);
    ASSERT_EQ(o.size(), 1u);
    EXPECT_EQ(o[0].off, 0u);
    EXPECT_EQ(o[0].len, 200 + G);
    EXPECT_EQ(o[0].count, 2u);
    o = coalesce({{0, 100}, {101 + G, 100}}, M, G);
    ASSERT_EQ(o.size(), 2u);
    o = coalesce({{5, 3 * M + 7}}, M, G);
    ASSERT_EQ(o.size(), 4u);
    EXPECT_EQ(o[3].off + o[3].len, 5 + 3 * M + 7);
    o = coalesce({{0, 10}, {0, 0}, {4, 3}, {20, 5}}, M, G);
    ASSERT_EQ(o.size(), 1u);
    EXPECT_EQ(o[0].len, 25u);
    o = coalesce({}, M, G);
    EXPECT_TRUE(o.empty());
}

TEST(RangeCoalesce, RejectsBadInputAndShortOutput) {
    const IoRange unsorted[2] = {{100, 1}, {50, 1}};
    CoalescedRange out[4];
    uint32_t n = 0;
    EXPECT_EQ(range_coalesce(unsorted, 2, 1000, 10, out, 4, &n), FileIoStatus::kBadArg);
    const IoRange ok[3] = {{0, 1}, {1000, 1}, {5000, 1}};
    EXPECT_EQ(range_coalesce(ok, 3, 0, 0, out, 4, &n), FileIoStatus::kBadArg);
    EXPECT_EQ(range_coalesce(ok, 3, 100, 100, out, 4, &n), FileIoStatus::kBadArg);
    EXPECT_EQ(range_coalesce(ok, 3, 100, 10, out, 2, &n), FileIoStatus::kTooLarge);
    EXPECT_EQ(range_coalesce(ok, 3, 100, 10, out, 3, &n), FileIoStatus::kOk);
    EXPECT_EQ(n, 3u);
}

TEST(RangeCoalesce, PropertyRandom) {
    const uint64_t iters = env_u64("BOLT_FIO_FUZZ_ITERS", 400);
    std::mt19937_64 rng(env_u64("BOLT_FIO_SEED", 42));
    for (uint64_t it = 0; it < iters; ++it) {
        const uint64_t max_gap = rng() % 64;
        const uint64_t max_req = max_gap + 1 + rng() % 512;
        const uint32_t n = static_cast<uint32_t>(rng() % 60);
        std::vector<IoRange> in(n);
        uint64_t pos = 0;
        for (IoRange& r : in) {
            pos += rng() % 3 == 0 ? 0 : rng() % 200;
            r.off = pos >= 50 && rng() % 4 == 0 ? pos - rng() % 50 : pos;  // overlaps
            r.len = rng() % 5 == 0 ? 0 : 1 + rng() % (rng() % 8 == 0 ? 2000 : 120);
            pos = std::max(pos, r.off);
        }
        std::sort(in.begin(), in.end(),
                  [](const IoRange& a, const IoRange& b) { return a.off < b.off; });
        SCOPED_TRACE("iter " + std::to_string(it));
        check_coalesce(in, max_req, max_gap, coalesce(in, max_req, max_gap));
        if (HasFatalFailure()) return;
    }
}

// --- batched reads -----------------------------------------------------------------

TEST(ReadBatch, FuzzMatchesSerialReads) {
    const std::string path = tmp_path("batch");
    const size_t n = (6u << 20) + 4321;
    const std::vector<uint8_t> bytes = random_bytes(n, 6);
    write_file(path, bytes);
    const uint64_t iters = env_u64("BOLT_FIO_FUZZ_ITERS", 400) / 8 + 1;
    const uint32_t workers[4] = {0, 1, 4, 11};
    for (uint32_t w : workers) {
        ReadPool* pool = nullptr;
        ASSERT_EQ(read_pool_create(w, &pool), FileIoStatus::kOk);
        ASSERT_EQ(read_pool_workers(pool), w);
        for (int direct = 0; direct < 2; ++direct) {
            ReadFile f{};
            ASSERT_EQ(file_open_read(path.c_str(), direct != 0, &f), FileIoStatus::kOk);
            std::mt19937_64 rng(100u + w * 2u + static_cast<unsigned>(direct));
            for (uint64_t it = 0; it < iters; ++it) {
                const uint32_t k = 1 + static_cast<uint32_t>(rng() % 64);
                std::vector<ReadReq> reqs(k);
                std::vector<AlignedBuf*> bufs;
                for (ReadReq& r : reqs) {
                    const uint64_t u = f.io_unit;
                    r = ReadReq{};
                    r.aligned = 1;
                    r.off = (rng() % (n / u + 2)) * u;
                    r.len = ((rng() % 40) + 1) * u;
                    bufs.push_back(new AlignedBuf(r.len, u));
                    r.dst = bufs.back()->p;
                }
                ASSERT_EQ(file_read_batch(rng() % 5 == 0 ? nullptr : pool, f, reqs.data(), k),
                          FileIoStatus::kOk);
                AlignedBuf serial(41 * f.io_unit, f.io_unit);
                for (const ReadReq& r : reqs) {
                    uint64_t got = 0;
                    ASSERT_EQ(r.status, FileIoStatus::kOk);
                    ASSERT_EQ(file_pread_aligned(f, r.off, serial.p, r.len, &got),
                              FileIoStatus::kOk);
                    ASSERT_EQ(r.got, got);
                    ASSERT_EQ(std::memcmp(r.dst, serial.p, got), 0);
                    ASSERT_EQ(std::memcmp(r.dst, bytes.data() + r.off, got), 0);
                }
                for (AlignedBuf* b : bufs) delete b;
            }
            file_close(&f);
        }
        read_pool_destroy(pool);
    }
    std::remove(path.c_str());
}

TEST(ReadBatch, FailedRequestReportedPerRequest) {
    const std::string path = tmp_path("batchfail");
    write_file(path, random_bytes(1u << 16, 8));
    ReadFile f{};
    ASSERT_EQ(file_open_read(path.c_str(), false, &f), FileIoStatus::kOk);
    ReadPool* pool = nullptr;
    ASSERT_EQ(read_pool_create(2, &pool), FileIoStatus::kOk);
    AlignedBuf buf(f.io_unit * 2, f.io_unit);
    ReadReq reqs[2]{};
    reqs[0].off = 0; reqs[0].len = f.io_unit; reqs[0].dst = buf.p; reqs[0].aligned = 1;
    reqs[1].off = 3; reqs[1].len = f.io_unit; reqs[1].dst = buf.p + f.io_unit;
    reqs[1].aligned = 1;
    EXPECT_EQ(file_read_batch(pool, f, reqs, 2), FileIoStatus::kMisaligned);
    EXPECT_EQ(reqs[0].status, FileIoStatus::kOk);
    EXPECT_EQ(reqs[1].status, FileIoStatus::kMisaligned);
    read_pool_destroy(pool);
    read_pool_destroy(nullptr);
    ReadPool* too_many = nullptr;
    EXPECT_EQ(read_pool_create(kReadPoolMaxWorkers + 1, &too_many), FileIoStatus::kTooLarge);
    file_close(&f);
    std::remove(path.c_str());
}

// --- cold throughput (reported; gated with BOLT_PERF_GATE=1) ---------------------

// Write with the direct writer, then evict, so the reads below are cold.
void write_bench_file(const std::string& path, uint64_t bytes) {
    AlignedWriterConfig cfg = aligned_writer_config_default();
    AlignedWriter w{};
    ASSERT_EQ(aligned_writer_open(path.c_str(), cfg, &w), AlignedWriteStatus::kOk);
    const std::vector<uint8_t> chunk = random_bytes(1u << 20, 11);
    for (uint64_t done = 0; done < bytes; done += chunk.size())
        ASSERT_EQ(aligned_writer_append(&w, chunk.data(), chunk.size()), AlignedWriteStatus::kOk);
    ASSERT_EQ(aligned_writer_finish(&w, true, nullptr), AlignedWriteStatus::kOk);
}

double cold_seq_gbps(const std::string& path, uint64_t bytes, uint32_t threads) {
    EXPECT_EQ(file_evict_cache(path.c_str()), FileIoStatus::kOk);
    ReadFile f{};
    EXPECT_EQ(file_open_read(path.c_str(), true, &f), FileIoStatus::kOk);
    ReadPool* pool = nullptr;
    EXPECT_EQ(read_pool_create(threads - 1, &pool), FileIoStatus::kOk);
    const uint64_t req = 1u << 20;
    const uint32_t per_batch = threads * 4;
    std::vector<AlignedBuf*> bufs;
    for (uint32_t i = 0; i < per_batch; ++i) bufs.push_back(new AlignedBuf(req, f.io_unit));
    std::vector<ReadReq> reqs(per_batch);
    uint64_t total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t off = 0; off < bytes; off += req * per_batch) {
        uint32_t k = 0;
        for (; k < per_batch && off + k * req < bytes; ++k) {
            reqs[k] = ReadReq{};
            reqs[k].off = off + k * req; reqs[k].len = req; reqs[k].dst = bufs[k]->p;
            reqs[k].aligned = 1;
        }
        EXPECT_EQ(file_read_batch(pool, f, reqs.data(), k), FileIoStatus::kOk);
        for (uint32_t i = 0; i < k; ++i) total += reqs[i].got;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_EQ(total, bytes);
    for (AlignedBuf* b : bufs) delete b;
    read_pool_destroy(pool);
    file_close(&f);
    return static_cast<double>(total) / s / 1e9;
}

TEST(FileIoBench, ColdAlignedPreadSequential) {
    const uint64_t bytes = env_u64("BOLT_FIO_BENCH_MB", 256) << 20;
    const uint32_t wide = static_cast<uint32_t>(env_u64("BOLT_FIO_THREADS", 12));
    const char* given = std::getenv("BOLT_FIO_BENCH_FILE");
    const bool own = given == nullptr || *given == '\0';
    const std::string path = own ? tmp_path("bench") : std::string(given);
    if (own) write_bench_file(path, bytes);
    if (HasFatalFailure()) return;
    const double one = cold_seq_gbps(path, bytes, 1);
    const double many = cold_seq_gbps(path, bytes, wide);
    std::printf("[fio] cold 1 MiB aligned direct pread, %llu MiB: 1 thread %.2f GB/s, "
                "%u threads %.2f GB/s\n",
                static_cast<unsigned long long>(bytes >> 20), one, wide, many);
    MappedFile m{};
    ASSERT_EQ(file_evict_cache(path.c_str()), FileIoStatus::kOk);
    ASSERT_EQ(mapped_file_open(path.c_str(), &m), FileIoStatus::kOk);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t sum = 0;
    const uint64_t ahead = 8u << 20;
    ASSERT_EQ(mapped_advise(m, 0, m.size, Advice::kSequential), FileIoStatus::kOk);
    for (uint64_t off = 0; off < m.size; off += m.page) {
        if (off % ahead == 0) (void)mapped_advise(m, off, 2 * ahead, Advice::kWillNeed);
        sum += m.base[off];
    }
    const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[fio] cold mmap + readahead touch, 1 thread: %.2f GB/s (sum %llu)\n",
                static_cast<double>(m.size) / ms / 1e9, static_cast<unsigned long long>(sum));
    mapped_file_close(&m);
    if (own) std::remove(path.c_str());
    if (env_u64("BOLT_PERF_GATE", 0) == 1) EXPECT_GE(one, 5.1);
}

}  // namespace
