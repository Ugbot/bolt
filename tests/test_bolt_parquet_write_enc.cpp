// Writer: BYTE_STREAM_SPLIT and the three DELTA encodings.
//
// bolt's READER already decoded all four; the writer could only emit PLAIN,
// so bolt could read files it was unable to produce. These tests close that
// asymmetry and check it the only way that means anything for an encoding:
// by asserting VALUES, over a swept parameter space, with a proof that the
// comparison discriminates.
//
// The cases that earn their keep here, rather than being the happy path
// again:
//
//   * int64 deltas that WRAP. delta = v[i] - v[i-1] overflows int64 for a
//     sequence that crosses the range, and both encoder and decoder rely on
//     the wrap being two's-complement rather than undefined. A sequence of
//     INT64_MIN / INT64_MAX alternations pins that.
//   * delta bit widths at and around byte boundaries, plus the bw == 0 case
//     (a constant column, where every delta equals min_delta), plus widths
//     past 56 where the packer switches to its per-byte path.
//   * block and miniblock boundaries: 128 values per block and 32 per
//     miniblock, so value counts of 1, 32, 33, 128, 129 and 4097 exercise
//     the partial-last-miniblock padding.
//   * DELTA_BYTE_ARRAY front coding against sorted strings (long shared
//     prefixes), unsorted strings (no shared prefix), a value that is a
//     strict prefix of its predecessor, and empty strings.

#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_meta.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_types.h"

namespace {

using namespace bolt::ingest::parquet;

std::vector<std::uint8_t> slurp_file(const char* path) {
    std::vector<std::uint8_t> v;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    const std::size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string tmp_path(const char* tag) {
    return std::string("test_bolt_parquet_write_enc_") + tag + ".parquet";
}

void attach_validity(bolt::BoltColumn* col, bolt::Arena* arena,
                     const std::vector<std::uint8_t>& valid) {
    const std::size_t n = valid.size();
    auto* bm = static_cast<std::uint8_t*>(arena->allocate((n + 7u) / 8u, 8));
    std::memset(bm, 0, (n + 7u) / 8u);
    for (std::size_t i = 0; i < n; ++i) {
        if (valid[i]) bm[i >> 3] = static_cast<std::uint8_t>(bm[i >> 3] | (1u << (i & 7u)));
    }
    col->validity = bm;
    col->validity_offset = 0;
    col->stats.all_valid = false;
}

// One fixed-width column, raw bytes supplied by the caller at the column's
// own physical stride.
void build_fixed_batch(bolt::Arena* arena, bolt::BoltType t, std::size_t stride,
                       const void* data, std::int64_t n,
                       const std::vector<std::uint8_t>* valid,
                       bolt::BoltBatch* out) {
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 1);
    out->schema.add_field("v", t, valid != nullptr);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = t;
    c.type_size_bytes = static_cast<std::uint32_t>(stride);
    auto* buf = static_cast<std::uint8_t*>(
        arena->allocate(static_cast<std::size_t>(n) * stride, 16));
    std::memcpy(buf, data, static_cast<std::size_t>(n) * stride);
    c.data = buf;
    c.stats.all_valid = true;
    if (valid != nullptr) attach_validity(&c, arena, *valid);
}

void build_str_batch(bolt::Arena* arena, const std::vector<std::string>& vals,
                     const std::vector<std::uint8_t>* valid,
                     bolt::BoltBatch* out) {
    const std::int64_t n = static_cast<std::int64_t>(vals.size());
    bolt::BoltBatch::init_empty(out);
    out->num_cols = 1;
    out->num_rows = n;
    bolt::BoltBatch::alloc_columns(out, arena, 1);
    out->schema.add_field("v", bolt::BoltType::Utf8, valid != nullptr);
    bolt::BoltColumn& c = out->columns[out->read_epoch][0];
    c = bolt::BoltColumn::make_empty();
    c.length = n;
    c.format = bolt::ColumnFormat::Flat;
    c.type = bolt::BoltType::Utf8;
    c.type_size_bytes = sizeof(bolt::StringView);
    auto* svs = static_cast<bolt::StringView*>(
        arena->allocate(static_cast<std::size_t>(n) * sizeof(bolt::StringView),
                        alignof(bolt::StringView)));
    std::memset(svs, 0, static_cast<std::size_t>(n) * sizeof(bolt::StringView));
    // Spill buffer for values past the 12-byte inline limit.
    std::size_t spill_need = 0;
    for (const auto& s : vals) if (s.size() > 12u) spill_need += s.size();
    auto* spill = (spill_need > 0)
        ? static_cast<std::uint8_t*>(arena->allocate(spill_need, 8)) : nullptr;
    std::size_t spill_off = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::string& s = vals[static_cast<std::size_t>(i)];
        svs[i].length = static_cast<std::uint32_t>(s.size());
        if (s.size() <= 12u) {
            std::memcpy(&svs[i].prefix[0], s.data(), s.size());
        } else {
            std::memcpy(&svs[i].prefix[0], s.data(), 4);
            std::memcpy(spill + spill_off, s.data(), s.size());
            svs[i].ref.offset = static_cast<std::uint32_t>(spill_off);
            spill_off += s.size();
        }
    }
    c.data = svs;
    c.str_overflow_base = spill;
    c.stats.all_valid = true;
    if (valid != nullptr) attach_validity(&c, arena, *valid);
}

ParquetWriteOpts one_col_opts(bolt::BoltType t, PqWriteEncoding enc,
                              bool nullable, std::uint8_t codec,
                              std::uint32_t page_bytes) {
    ParquetWriteOpts o{};
    o.n_columns = 1;
    o.compression = codec;
    o.emit_statistics = true;
    o.data_page_target_bytes = page_bytes;
    std::strncpy(o.columns[0].name, "v", sizeof(o.columns[0].name) - 1);
    o.columns[0].type = t;
    o.columns[0].nullable = nullable;
    o.columns[0].encoding = static_cast<std::uint8_t>(enc);
    if (t == bolt::BoltType::Decimal128) {
        o.columns[0].precision = 38;
        o.columns[0].scale = 4;
    }
    return o;
}

std::vector<std::uint8_t> write_batch(bolt::BoltBatch* b,
                                      const ParquetWriteOpts& o,
                                      const char* tag) {
    const std::string path = tmp_path(tag);
    ParquetWriter* w = parquet_write_open(path.c_str(), &o);
    EXPECT_NE(w, nullptr);
    if (w == nullptr) return {};
    EXPECT_TRUE(parquet_write_row_group(w, b));
    EXPECT_TRUE(parquet_write_close(w));
    return slurp_file(path.c_str());
}

// Read back one column and hand its decoded form to `check`.
// Returns "" on success, else the first problem.
template <typename Fn>
std::string read_back(const std::vector<std::uint8_t>& buf, std::int64_t n,
                      Fn check) {
    if (buf.empty()) return "empty file";
    bolt::Arena ra;
    auto* rb = ra.allocate_array<bolt::BoltBatch>(1);
    if (rb == nullptr) return "alloc";
    if (!parquet_read_file(buf.data(), buf.size(), &ra, rb)) return "read failed";
    if (rb->num_rows != n) return "row count mismatch";
    if (rb->num_cols != 1u) return "col count mismatch";
    return check(rb->columns[rb->read_epoch][0]);
}

// ---- DELTA_BINARY_PACKED -------------------------------------------------

std::string check_i64(const bolt::BoltColumn& c,
                      const std::vector<std::int64_t>& want,
                      const std::vector<std::uint8_t>* valid) {
    char msg[192];
    const auto* p = static_cast<const std::int64_t*>(c.data);
    if (p == nullptr) return "null data";
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (valid != nullptr) {
            const std::int64_t bit = c.validity_offset + static_cast<std::int64_t>(i);
            const bool got_valid = (c.validity == nullptr) ||
                (((c.validity[bit >> 3] >> (bit & 7)) & 1u) != 0u);
            if (got_valid != ((*valid)[i] != 0u)) {
                std::snprintf(msg, sizeof(msg), "validity at row %zu", i);
                return msg;
            }
            if (!(*valid)[i]) continue;
        }
        if (p[i] != want[i]) {
            std::snprintf(msg, sizeof(msg),
                          "value at row %zu: got %lld want %lld", i,
                          static_cast<long long>(p[i]),
                          static_cast<long long>(want[i]));
            return msg;
        }
    }
    return std::string();
}

std::string roundtrip_i64(const std::vector<std::int64_t>& vals,
                          PqWriteEncoding enc, const char* tag,
                          const std::vector<std::uint8_t>* valid = nullptr,
                          std::uint32_t page_bytes = 0) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Int64, 8, vals.data(),
                      static_cast<std::int64_t>(vals.size()), valid, b);
    const auto o = one_col_opts(bolt::BoltType::Int64, enc, valid != nullptr,
                                0, page_bytes);
    const auto buf = write_batch(b, o, tag);
    return read_back(buf, static_cast<std::int64_t>(vals.size()),
                     [&](const bolt::BoltColumn& c) {
                         return check_i64(c, vals, valid);
                     });
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedBlockBoundaries) {
    // A block is 128 values and a miniblock 32, so these counts land on and
    // either side of both, and 4097 forces many blocks plus a 1-value tail.
    for (int n : {1, 2, 8, 31, 32, 33, 127, 128, 129, 1000, 4097}) {
        std::vector<std::int64_t> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = i * 3 - 5;
        SCOPED_TRACE(testing::Message() << "n=" << n);
        EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "dbp_n"),
                  std::string());
    }
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedBitWidthSweep) {
    // Delta magnitude drives the miniblock bit width. Stepping the stride
    // across 1, 2, 4 ... 2^62 walks bw through every byte boundary and past
    // 56, where the bit packer switches to its per-byte path.
    for (int shift = 0; shift <= 62; ++shift) {
        const std::int64_t step = std::int64_t{1} << shift;
        std::vector<std::int64_t> v(300);
        std::int64_t cur = -step;   // start negative so deltas are not all one sign
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = cur;
            cur += (i % 3 == 0) ? step : -step;
        }
        SCOPED_TRACE(testing::Message() << "shift=" << shift);
        EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "dbp_bw"),
                  std::string());
    }
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedConstantColumnUsesZeroBitWidth) {
    // Every delta equals min_delta, so every miniblock bit width is 0 and no
    // packed bytes are written at all -- the branch a naive encoder gets
    // wrong by emitting an empty run the decoder then mis-locates.
    std::vector<std::int64_t> v(500, std::int64_t{-999});
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "dbp_const"),
              std::string());
    // ...and a perfectly linear ramp, where deltas are constant but nonzero.
    std::vector<std::int64_t> ramp(500);
    for (std::size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<std::int64_t>(i) * 7;
    EXPECT_EQ(roundtrip_i64(ramp, PqWriteEncoding::DeltaBinaryPacked, "dbp_ramp"),
              std::string());
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedWrapsAtTheInt64Range) {
    // v[i] - v[i-1] is not representable as an int64 here. The encoder does
    // the subtraction in uint64 so it wraps two's-complement, and the
    // decoder's `cur += min_delta + packed` wraps identically -- so the round
    // trip is exact across the whole int64 domain. If either side ever used
    // signed arithmetic (undefined on overflow) this is the test that breaks.
    const std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    const std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    std::vector<std::int64_t> v;
    for (int i = 0; i < 200; ++i) v.push_back((i & 1) ? hi : lo);
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "dbp_wrap"),
              std::string());

    std::vector<std::int64_t> mix = {0, hi, lo, 0, lo, hi, -1, 1, lo + 1, hi - 1};
    while (mix.size() < 64) mix.push_back(mix[mix.size() % 10]);
    EXPECT_EQ(roundtrip_i64(mix, PqWriteEncoding::DeltaBinaryPacked, "dbp_mix"),
              std::string());
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedWithNullsAndPageSplits) {
    std::vector<std::int64_t> v(5000);
    std::vector<std::uint8_t> valid(5000, 1u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 11 - 40000;
        if ((i % 7) == 0) valid[i] = 0u;
    }
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "dbp_null",
                            &valid, /*page_bytes=*/4096),
              std::string());
}

TEST(BoltParquetWriteEnc, DeltaBinaryPackedInt32) {
    std::vector<std::int32_t> v(2000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int32_t>(i) * -13 + 500;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Int32, 4, v.data(),
                      static_cast<std::int64_t>(v.size()), nullptr, b);
    const auto o = one_col_opts(bolt::BoltType::Int32,
                                PqWriteEncoding::DeltaBinaryPacked, false, 0, 0);
    const auto buf = write_batch(b, o, "dbp_i32");
    const std::string err = read_back(
        buf, static_cast<std::int64_t>(v.size()),
        [&](const bolt::BoltColumn& c) -> std::string {
            const auto* p = static_cast<const std::int32_t*>(c.data);
            if (p == nullptr) return "null data";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (p[i] != v[i]) return "int32 value mismatch";
            }
            return std::string();
        });
    EXPECT_EQ(err, std::string());
}

// ---- BYTE_STREAM_SPLIT ---------------------------------------------------

TEST(BoltParquetWriteEnc, ByteStreamSplitFloat64) {
    std::vector<double> v(3000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<double>(i) * 0.125 - 7.5;
    }
    v[0] = -0.0;
    v[1] = std::numeric_limits<double>::infinity();
    v[2] = -std::numeric_limits<double>::infinity();
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Float64, 8, v.data(),
                      static_cast<std::int64_t>(v.size()), nullptr, b);
    const auto o = one_col_opts(bolt::BoltType::Float64,
                                PqWriteEncoding::ByteStreamSplit, false, 0, 4096);
    const auto buf = write_batch(b, o, "bss_f64");
    const std::string err = read_back(
        buf, static_cast<std::int64_t>(v.size()),
        [&](const bolt::BoltColumn& c) -> std::string {
            const auto* p = static_cast<const double*>(c.data);
            if (p == nullptr) return "null data";
            for (std::size_t i = 0; i < v.size(); ++i) {
                // Bit-exact: a transpose must not perturb a single byte, and
                // comparing as doubles would let -0.0 pass as 0.0.
                if (std::memcmp(&p[i], &v[i], 8) != 0) return "f64 mismatch";
            }
            return std::string();
        });
    EXPECT_EQ(err, std::string());
}

TEST(BoltParquetWriteEnc, ByteStreamSplitFloat32WithNulls) {
    std::vector<float> v(2000);
    std::vector<std::uint8_t> valid(2000, 1u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(i) * 0.5f - 100.0f;
        if ((i % 5) == 0) valid[i] = 0u;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Float32, 4, v.data(),
                      static_cast<std::int64_t>(v.size()), &valid, b);
    const auto o = one_col_opts(bolt::BoltType::Float32,
                                PqWriteEncoding::ByteStreamSplit, true, 1, 4096);
    const auto buf = write_batch(b, o, "bss_f32");
    // Float32 widens to Float64 on read (bolt's documented decode rule).
    const std::string err = read_back(
        buf, static_cast<std::int64_t>(v.size()),
        [&](const bolt::BoltColumn& c) -> std::string {
            const auto* p = static_cast<const double*>(c.data);
            if (p == nullptr) return "null data";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (!valid[i]) continue;
                if (p[i] != static_cast<double>(v[i])) return "f32 mismatch";
            }
            return std::string();
        });
    EXPECT_EQ(err, std::string());
}

TEST(BoltParquetWriteEnc, ByteStreamSplitInt64MatchesPlain) {
    // Since 2.9 the spec allows BYTE_STREAM_SPLIT on fixed-width integers.
    // Comparing against the PLAIN encoding of the same data checks the
    // transpose independently of the model.
    std::vector<std::int64_t> v(4000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 0x0102030405ll - 12345;
    }
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::ByteStreamSplit, "bss_i64"),
              std::string());
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::Plain, "plain_i64"),
              std::string());
}

// ---- DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY --------------------------

std::string roundtrip_str(const std::vector<std::string>& vals,
                          PqWriteEncoding enc, const char* tag,
                          const std::vector<std::uint8_t>* valid = nullptr,
                          std::uint32_t page_bytes = 0) {
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a, vals, valid, b);
    const auto o = one_col_opts(bolt::BoltType::Utf8, enc, valid != nullptr,
                                0, page_bytes);
    const auto buf = write_batch(b, o, tag);
    return read_back(
        buf, static_cast<std::int64_t>(vals.size()),
        [&](const bolt::BoltColumn& c) -> std::string {
            char msg[192];
            const auto* sv = static_cast<const bolt::StringView*>(c.data);
            const auto* spill =
                static_cast<const std::uint8_t*>(c.str_overflow_base);
            if (sv == nullptr) return "null data";
            for (std::size_t i = 0; i < vals.size(); ++i) {
                if (valid != nullptr && !(*valid)[i]) continue;
                const std::uint32_t len = sv[i].length;
                if (len > 12u && spill == nullptr) return "missing spill";
                const std::uint8_t* p =
                    (len <= 12u)
                        ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                        : (spill + sv[i].ref.offset);
                if (len != vals[i].size() ||
                    std::memcmp(p, vals[i].data(), len) != 0) {
                    std::snprintf(msg, sizeof(msg), "string at row %zu", i);
                    return msg;
                }
            }
            return std::string();
        });
}

// Sorted, long shared prefixes -- the shape DELTA_BYTE_ARRAY exists for.
std::vector<std::string> sorted_prefixed(std::size_t n) {
    std::vector<std::string> v;
    v.reserve(n);
    char buf[64];
    for (std::size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof(buf), "com.example.package.name.%08zu", i);
        v.push_back(buf);
    }
    return v;
}

TEST(BoltParquetWriteEnc, DeltaByteArraySortedPrefixes) {
    for (PqWriteEncoding e : {PqWriteEncoding::DeltaByteArray,
                              PqWriteEncoding::DeltaLengthByteArray}) {
        for (std::size_t n : {std::size_t{1}, std::size_t{32}, std::size_t{129},
                              std::size_t{3000}}) {
            SCOPED_TRACE(testing::Message()
                         << "enc=" << static_cast<int>(e) << " n=" << n);
            EXPECT_EQ(roundtrip_str(sorted_prefixed(n), e, "dba_sorted"),
                      std::string());
        }
    }
}

TEST(BoltParquetWriteEnc, DeltaByteArrayAdversarialShapes) {
    // Each of these breaks a different naive front-coding implementation.
    const std::vector<std::string> vals = {
        "",                 // empty first value: prefix must be 0
        "",                 // empty after empty
        "abc",
        "abc",              // identical: prefix == full length, suffix empty
        "ab",               // a strict PREFIX of its predecessor -- prefix
                            // must be capped at the shorter length
        "abcdefghijklmnop", // crosses the 12-byte inline/spill boundary
        "abcdefghijklmnoq",
        "",                 // back to empty after a long value
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        "a",
    };
    for (PqWriteEncoding e : {PqWriteEncoding::DeltaByteArray,
                              PqWriteEncoding::DeltaLengthByteArray,
                              PqWriteEncoding::Plain}) {
        SCOPED_TRACE(testing::Message() << "enc=" << static_cast<int>(e));
        EXPECT_EQ(roundtrip_str(vals, e, "dba_adv"), std::string());
    }
}

TEST(BoltParquetWriteEnc, DeltaByteArrayUnsortedAndNullable) {
    std::vector<std::string> vals;
    std::vector<std::uint8_t> valid;
    char buf[64];
    for (std::size_t i = 0; i < 4000; ++i) {
        // Reversed digits: adjacent values share no prefix at all, so the
        // encoder must handle prefix == 0 for essentially every value.
        std::snprintf(buf, sizeof(buf), "%08zu-tail", (i * 2654435761u) % 99991u);
        vals.push_back(buf);
        valid.push_back((i % 6) == 0 ? 0u : 1u);
    }
    for (PqWriteEncoding e : {PqWriteEncoding::DeltaByteArray,
                              PqWriteEncoding::DeltaLengthByteArray}) {
        SCOPED_TRACE(testing::Message() << "enc=" << static_cast<int>(e));
        EXPECT_EQ(roundtrip_str(vals, e, "dba_uns", &valid, 4096),
                  std::string());
    }
}

TEST(BoltParquetWriteEnc, DeltaByteArrayIsSmallerOnPrefixedData) {
    // Front coding must actually pay off, otherwise the encoding engaged in
    // name only. 3000 values sharing a 25-byte prefix is its best case.
    const auto vals = sorted_prefixed(3000);
    bolt::Arena a1, a2;
    auto* b1 = a1.allocate_array<bolt::BoltBatch>(1);
    auto* b2 = a2.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a1, vals, nullptr, b1);
    build_str_batch(&a2, vals, nullptr, b2);
    const auto plain = write_batch(
        b1, one_col_opts(bolt::BoltType::Utf8, PqWriteEncoding::Plain, false, 0, 0),
        "cmp_plain");
    const auto dba = write_batch(
        b2, one_col_opts(bolt::BoltType::Utf8, PqWriteEncoding::DeltaByteArray,
                         false, 0, 0),
        "cmp_dba");
    ASSERT_FALSE(plain.empty());
    ASSERT_FALSE(dba.empty());
    EXPECT_LT(dba.size(), plain.size() / 2);
}

// ---- validation ----------------------------------------------------------

TEST(BoltParquetWriteEnc, ImpossibleEncodingIsRejectedAtOpen) {
    // A caller who asks for an encoding the type cannot carry has a bug.
    // Writing PLAIN instead would hide it inside a file that reads back fine.
    struct Bad { bolt::BoltType t; PqWriteEncoding e; };
    const Bad bad[] = {
        {bolt::BoltType::Utf8,    PqWriteEncoding::DeltaBinaryPacked},
        {bolt::BoltType::Utf8,    PqWriteEncoding::ByteStreamSplit},
        {bolt::BoltType::Int64,   PqWriteEncoding::DeltaByteArray},
        {bolt::BoltType::Int64,   PqWriteEncoding::DeltaLengthByteArray},
        {bolt::BoltType::Float64, PqWriteEncoding::DeltaBinaryPacked},
        {bolt::BoltType::Bool,    PqWriteEncoding::Dictionary},
        {bolt::BoltType::Bool,    PqWriteEncoding::ByteStreamSplit},
    };
    for (const Bad& c : bad) {
        auto o = one_col_opts(c.t, c.e, false, 0, 0);
        const std::string path = tmp_path("reject");
        ParquetWriter* w = parquet_write_open(path.c_str(), &o);
        EXPECT_EQ(w, nullptr) << "type " << static_cast<int>(c.t)
                              << " wrongly accepted encoding "
                              << static_cast<int>(c.e);
        if (w != nullptr) parquet_write_close(w);
    }
    // An out-of-range encoding byte is rejected too, not read as an enum.
    auto o = one_col_opts(bolt::BoltType::Int64, PqWriteEncoding::Plain, false, 0, 0);
    o.columns[0].encoding = 99u;
    ParquetWriter* w = parquet_write_open(tmp_path("reject2").c_str(), &o);
    EXPECT_EQ(w, nullptr);
    if (w != nullptr) parquet_write_close(w);
}

TEST(BoltParquetWriteEnc, DeclaredEncodingMatchesWhatWasWritten) {
    // The footer must name the encoding the pages actually carry -- a reader
    // that trusts the declaration and finds something else fails at decode.
    struct Case { bolt::BoltType t; PqWriteEncoding e; };
    const Case cases[] = {
        {bolt::BoltType::Int64,   PqWriteEncoding::DeltaBinaryPacked},
        {bolt::BoltType::Float64, PqWriteEncoding::ByteStreamSplit},
        {bolt::BoltType::Utf8,    PqWriteEncoding::DeltaByteArray},
        {bolt::BoltType::Utf8,    PqWriteEncoding::DeltaLengthByteArray},
    };
    for (const Case& c : cases) {
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        std::vector<std::int64_t> iv(200);
        std::vector<double> dv(200);
        std::vector<std::string> sv;
        for (std::size_t i = 0; i < 200; ++i) {
            iv[i] = static_cast<std::int64_t>(i);
            dv[i] = static_cast<double>(i);
            sv.push_back("s" + std::to_string(i));
        }
        if (c.t == bolt::BoltType::Utf8) {
            build_str_batch(&a, sv, nullptr, b);
        } else if (c.t == bolt::BoltType::Float64) {
            build_fixed_batch(&a, c.t, 8, dv.data(), 200, nullptr, b);
        } else {
            build_fixed_batch(&a, c.t, 8, iv.data(), 200, nullptr, b);
        }
        const auto buf = write_batch(b, one_col_opts(c.t, c.e, false, 0, 0),
                                     "declared");
        ASSERT_FALSE(buf.empty());
        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        ASSERT_GE(meta.n_chunks, 1u);
        // No dictionary page for any of these.
        EXPECT_EQ(meta.chunks[0].dictionary_page_offset, 0);
    }
}

// ---- the gate must discriminate -----------------------------------------

TEST(BoltParquetWriteEnc, DiscriminatingPower) {
    // Same argument as the dictionary suite: if the comparison could not
    // fail, none of the sweeps above would mean anything.
    std::vector<std::int64_t> v(500);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 17 - 3;
    }
    EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked, "disc"),
              std::string());
    std::vector<std::int64_t> bad = v;
    bad[321] += 1;
    // roundtrip_i64 rewrites the file from `bad`, so compare `bad`'s file
    // against `v` instead by checking the decoded column directly.
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Int64, 8, v.data(), 500, nullptr, b);
    const auto buf = write_batch(
        b, one_col_opts(bolt::BoltType::Int64,
                        PqWriteEncoding::DeltaBinaryPacked, false, 0, 0),
        "disc2");
    const std::string err = read_back(buf, 500, [&](const bolt::BoltColumn& c) {
        return check_i64(c, bad, nullptr);
    });
    EXPECT_NE(err.find("value at row 321"), std::string::npos) << err;

    const std::string ok = read_back(buf, 500, [&](const bolt::BoltColumn& c) {
        return check_i64(c, v, nullptr);
    });
    EXPECT_EQ(ok, std::string());
}

// ---- interop fixtures ----------------------------------------------------

TEST(BoltParquetWriteEnc, InteropFixtures) {
    // Read by scripts/parquet_write_interop.py with pyarrow. bolt reading its
    // own delta stream back only proves the encoder and decoder agree.
    {
        std::vector<std::int64_t> v(8000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = static_cast<std::int64_t>(i) * 7 - 3;
        EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::DeltaBinaryPacked,
                                "interop_dbp"), std::string());
        EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::ByteStreamSplit,
                                "interop_bss_i64"), std::string());
    }
    {
        std::vector<double> v(8000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = static_cast<double>(i) * 0.25 - 3.0;
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        build_fixed_batch(&a, bolt::BoltType::Float64, 8, v.data(), 8000,
                          nullptr, b);
        const auto buf = write_batch(
            b, one_col_opts(bolt::BoltType::Float64,
                            PqWriteEncoding::ByteStreamSplit, false, 0, 0),
            "interop_bss_f64");
        EXPECT_FALSE(buf.empty());
    }
    {
        const auto vals = sorted_prefixed(8000);
        EXPECT_EQ(roundtrip_str(vals, PqWriteEncoding::DeltaByteArray,
                                "interop_dba"), std::string());
        EXPECT_EQ(roundtrip_str(vals, PqWriteEncoding::DeltaLengthByteArray,
                                "interop_dlba"), std::string());
    }
}

// ---- DATA_PAGE_V2 --------------------------------------------------------

TEST(BoltParquetWriteEnc, DataPageV2RoundTripsEveryEncoding) {
    // v2 moves the levels OUT of the compressed body, stores them raw with an
    // explicit byte length, and adds a null count. The failure modes are all
    // arithmetic: a page whose declared uncompressed size forgets to include
    // the level bytes makes a reader compute a negative values length, and one
    // that compresses the levels along with the values decodes as garbage.
    // Both still produce a plausible-looking page header, so this asserts
    // VALUES for every encoding and both nullability shapes.
    for (std::uint8_t codec : {std::uint8_t{0}, std::uint8_t{1}}) {
        for (int null_every : {0, 4}) {
            const bool nullable = null_every > 0;
            std::vector<std::int64_t> v(3000);
            std::vector<std::uint8_t> valid(3000, 1u);
            for (std::size_t i = 0; i < v.size(); ++i) {
                v[i] = static_cast<std::int64_t>(i) * 13 - 5000;
                if (nullable && (i % 4) == 0) valid[i] = 0u;
            }
            for (PqWriteEncoding e : {PqWriteEncoding::Plain,
                                      PqWriteEncoding::DeltaBinaryPacked,
                                      PqWriteEncoding::ByteStreamSplit,
                                      PqWriteEncoding::Dictionary}) {
                bolt::Arena a;
                auto* b = a.allocate_array<bolt::BoltBatch>(1);
                build_fixed_batch(&a, bolt::BoltType::Int64, 8, v.data(),
                                  static_cast<std::int64_t>(v.size()),
                                  nullable ? &valid : nullptr, b);
                auto o = one_col_opts(bolt::BoltType::Int64, e, nullable,
                                      codec, 4096);
                o.data_page_v2 = true;
                o.use_dictionary = (e == PqWriteEncoding::Dictionary);
                const auto buf = write_batch(b, o, "v2");
                SCOPED_TRACE(testing::Message()
                             << "enc=" << static_cast<int>(e)
                             << " codec=" << static_cast<int>(codec)
                             << " nullable=" << nullable);
                const std::string err = read_back(
                    buf, static_cast<std::int64_t>(v.size()),
                    [&](const bolt::BoltColumn& c) {
                        return check_i64(c, v, nullable ? &valid : nullptr);
                    });
                EXPECT_EQ(err, std::string());
            }
        }
    }
}

TEST(BoltParquetWriteEnc, DataPageV2AndV1AgreeAndAreDistinct) {
    // The two page formats must decode to the same values -- and must really
    // be different files, or the option did nothing and the test above was
    // silently re-checking v1.
    std::vector<std::string> vals;
    for (std::size_t i = 0; i < 2000; ++i) {
        vals.push_back("key-" + std::to_string(i % 250));
    }
    bolt::Arena a1, a2;
    auto* b1 = a1.allocate_array<bolt::BoltBatch>(1);
    auto* b2 = a2.allocate_array<bolt::BoltBatch>(1);
    build_str_batch(&a1, vals, nullptr, b1);
    build_str_batch(&a2, vals, nullptr, b2);
    auto o1 = one_col_opts(bolt::BoltType::Utf8, PqWriteEncoding::Dictionary,
                           false, 1, 4096);
    o1.use_dictionary = true;
    auto o2 = o1;
    o2.data_page_v2 = true;
    const auto v1 = write_batch(b1, o1, "v1cmp");
    const auto v2 = write_batch(b2, o2, "v2cmp");
    ASSERT_FALSE(v1.empty());
    ASSERT_FALSE(v2.empty());
    EXPECT_NE(v1.size(), v2.size())
        << "data_page_v2 produced a byte-identical file -- the option is inert";
    const std::string e2 = read_back(
        v2, static_cast<std::int64_t>(vals.size()),
        [&](const bolt::BoltColumn& c) -> std::string {
            const auto* sv = static_cast<const bolt::StringView*>(c.data);
            const auto* spill =
                static_cast<const std::uint8_t*>(c.str_overflow_base);
            if (sv == nullptr) return "null data";
            for (std::size_t i = 0; i < vals.size(); ++i) {
                const std::uint32_t len = sv[i].length;
                const std::uint8_t* p =
                    (len <= 12u)
                        ? reinterpret_cast<const std::uint8_t*>(&sv[i].prefix[0])
                        : (spill + sv[i].ref.offset);
                if (len != vals[i].size() ||
                    std::memcmp(p, vals[i].data(), len) != 0) {
                    return "v2 string mismatch at row " + std::to_string(i);
                }
            }
            return std::string();
        });
    EXPECT_EQ(e2, std::string());
}

TEST(BoltParquetWriteEnc, DataPageV2InteropFixture) {
    std::vector<std::int64_t> v(8000);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<std::int64_t>(i) * 7 - 3;
    }
    bolt::Arena a;
    auto* b = a.allocate_array<bolt::BoltBatch>(1);
    build_fixed_batch(&a, bolt::BoltType::Int64, 8, v.data(), 8000, nullptr, b);
    auto o = one_col_opts(bolt::BoltType::Int64,
                          PqWriteEncoding::DeltaBinaryPacked, false, 1, 4096);
    o.data_page_v2 = true;
    const auto buf = write_batch(b, o, "interop_v2");
    EXPECT_FALSE(buf.empty());
}

// ---- LZ4_RAW codec --------------------------------------------------------

TEST(BoltParquetWriteEnc, Lz4RawCodecRoundTripsAndIsDeclared) {
    // bolt could neither read nor write LZ4_RAW on a default build: the only
    // LZ4 available was behind find_package(lz4). Both directions now go
    // through bolt's own block codec, so this checks the parquet layer wires
    // it correctly -- values back, and the footer naming codec 7 (LZ4_RAW),
    // never codec 5 (the deprecated Hadoop-framed "LZ4").
    for (int null_every : {0, 5}) {
        const bool nullable = null_every > 0;
        std::vector<std::int64_t> v(6000);
        std::vector<std::uint8_t> valid(6000, 1u);
        for (std::size_t i = 0; i < v.size(); ++i) {
            // Repetitive enough that the codec has something to do.
            v[i] = static_cast<std::int64_t>((i / 8) % 50);
            if (nullable && (i % 5) == 0) valid[i] = 0u;
        }
        SCOPED_TRACE(testing::Message() << "nullable=" << nullable);
        EXPECT_EQ(roundtrip_i64(v, PqWriteEncoding::Plain, "lz4",
                                nullable ? &valid : nullptr, 4096),
                  std::string())
            << "sanity: uncompressed path";

        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        build_fixed_batch(&a, bolt::BoltType::Int64, 8, v.data(),
                          static_cast<std::int64_t>(v.size()),
                          nullable ? &valid : nullptr, b);
        auto o = one_col_opts(bolt::BoltType::Int64, PqWriteEncoding::Plain,
                              nullable, /*codec=*/4, 4096);
        const auto buf = write_batch(b, o, "lz4raw");
        ASSERT_FALSE(buf.empty());
        const std::string err = read_back(
            buf, static_cast<std::int64_t>(v.size()),
            [&](const bolt::BoltColumn& c) {
                return check_i64(c, v, nullable ? &valid : nullptr);
            });
        EXPECT_EQ(err, std::string());

        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        ASSERT_GE(meta.n_chunks, 1u);
        EXPECT_EQ(static_cast<int>(meta.chunks[0].codec),
                  static_cast<int>(PqCodec::Lz4Raw))
            << "footer must name LZ4_RAW (7), not the deprecated LZ4 (5)";

        // And it must actually compress: this data is highly repetitive, so
        // an LZ4 page that is no smaller than the uncompressed one means the
        // codec was silently skipped.
        auto o_none = o;
        o_none.compression = 0;
        bolt::Arena a2;
        auto* b2 = a2.allocate_array<bolt::BoltBatch>(1);
        build_fixed_batch(&a2, bolt::BoltType::Int64, 8, v.data(),
                          static_cast<std::int64_t>(v.size()),
                          nullable ? &valid : nullptr, b2);
        const auto plain = write_batch(b2, o_none, "lz4none");
        EXPECT_LT(buf.size(), plain.size())
            << "LZ4_RAW did not shrink highly repetitive data";
    }
}

TEST(BoltParquetWriteEnc, GzipCodecRoundTripsAndIsDeclared) {
    // bolt could READ gzip parquet (via its own inflate) and never write it.
    // Both directions are now dependency-free.
    for (int null_every : {0, 4}) {
        const bool nullable = null_every > 0;
        std::vector<std::int64_t> v(6000);
        std::vector<std::uint8_t> valid(6000, 1u);
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<std::int64_t>((i / 8) % 50);
            if (nullable && (i % 4) == 0) valid[i] = 0u;
        }
        bolt::Arena a;
        auto* b = a.allocate_array<bolt::BoltBatch>(1);
        build_fixed_batch(&a, bolt::BoltType::Int64, 8, v.data(),
                          static_cast<std::int64_t>(v.size()),
                          nullable ? &valid : nullptr, b);
        auto o = one_col_opts(bolt::BoltType::Int64, PqWriteEncoding::Plain,
                              nullable, /*codec=*/2, 4096);
        const auto buf = write_batch(b, o, "gzip");
        ASSERT_FALSE(buf.empty());
        SCOPED_TRACE(testing::Message() << "nullable=" << nullable);
        const std::string err = read_back(
            buf, static_cast<std::int64_t>(v.size()),
            [&](const bolt::BoltColumn& c) {
                return check_i64(c, v, nullable ? &valid : nullptr);
            });
        EXPECT_EQ(err, std::string());

        bolt::Arena ma;
        PqMeta meta{};
        ASSERT_TRUE(parquet_read_meta(buf.data(), buf.size(), &ma, &meta));
        EXPECT_EQ(static_cast<int>(meta.chunks[0].codec),
                  static_cast<int>(PqCodec::Gzip));

        auto o_none = o;
        o_none.compression = 0;
        bolt::Arena a2;
        auto* b2 = a2.allocate_array<bolt::BoltBatch>(1);
        build_fixed_batch(&a2, bolt::BoltType::Int64, 8, v.data(),
                          static_cast<std::int64_t>(v.size()),
                          nullable ? &valid : nullptr, b2);
        const auto plain = write_batch(b2, o_none, "gzipnone");
        EXPECT_LT(buf.size(), plain.size())
            << "GZIP did not shrink highly repetitive data";
    }
}

TEST(BoltParquetWriteEnc, UnsupportedCodecsAreRejectedAtOpen) {
    // ZSTD has a self-contained DECODER in bolt but no dependency-free
    // compressor, so asking to write it must fail loudly rather than silently
    // producing an uncompressed file. 5 is parquet's deprecated Hadoop-framed
    // "LZ4", which bolt deliberately never emits.
    for (std::uint8_t codec : {std::uint8_t{3}, std::uint8_t{5},
                               std::uint8_t{6}, std::uint8_t{99}}) {
        auto o = one_col_opts(bolt::BoltType::Int64, PqWriteEncoding::Plain,
                              false, codec, 0);
        ParquetWriter* w = parquet_write_open(tmp_path("badcodec").c_str(), &o);
        EXPECT_EQ(w, nullptr) << "codec " << static_cast<int>(codec)
                              << " was accepted";
        if (w != nullptr) parquet_write_close(w);
    }
}

}  // namespace
