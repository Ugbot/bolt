// test_bolt_wire_frame.cpp — B2: checksummed wire frames, the multi-frame
// container (index, rollup, footer), crash recovery, recorded alignment units.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_budget.h"
#include "bolt/bolt_column.h"
#include "bolt/wire/bolt_wire.h"
#include "bolt/wire/bolt_wire_frame.h"
#include "bolt/wire/bolt_wire_frame_file.h"

using namespace bolt;
using namespace bolt::wire;

namespace {

// 64 B-aligned scratch, like an mmap of a file.
struct Buf {
    std::vector<uint8_t> raw;
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit Buf(size_t bytes, uint8_t fill = 0) : raw(bytes + 64, fill) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(raw.data());
        p = raw.data() + ((64 - (a & 63)) & 63);
        n = bytes;
    }
};

void set_field(BoltBatch* b, uint32_t i, const char* name, BoltType t) {
    BoltField& f = b->schema.fields[i];
    std::memset(&f, 0, sizeof(f));
    f.set_name(name);
    f.type = t;
    f.nullable = true;
}

// Columns: i64 (nulls), f64 (NaN), utf8 (spilled), u32, decimal128 (no
// order), bool. `seed` shifts values so frames differ.
void build_batch(BoltBatch* b, Arena* a, int64_t n, int64_t seed) {
    BoltBatch::init_empty(b);
    ASSERT_TRUE(BoltBatch::alloc_columns(b, a, 6));
    b->num_rows = n;
    b->schema.num_fields = 6;
    const BoltType types[6] = {BoltType::Int64, BoltType::Float64, BoltType::Utf8,
                               BoltType::UInt32, BoltType::Decimal128, BoltType::Bool};
    const char* names[6] = {"i", "f", "s", "u", "d", "b"};
    uint8_t* v = static_cast<uint8_t*>(a->allocate(static_cast<size_t>((n + 7) / 8) + 1, 64));
    std::memset(v, 0, static_cast<size_t>((n + 7) / 8) + 1);
    for (int64_t r = 0; r < n; ++r) if (r % 5 != 2) v[r >> 3] |= uint8_t(1u << (r & 7));
    char* pool = static_cast<char*>(a->allocate(static_cast<size_t>(n) * 24 + 1, 64));
    uint32_t used = 0;
    for (uint32_t c = 0; c < 6; ++c) {
        set_field(b, c, names[c], types[c]);
        BoltColumn col = BoltColumn::make_flat_alloc(n, types[c], a);
        for (int64_t r = 0; r < n; ++r) {
            const int64_t x = (r * 7919 + seed * 31) % 1000 - 500;
            switch (c) {
                case 0: static_cast<int64_t*>(col.data)[r] = x; break;
                case 1: static_cast<double*>(col.data)[r] =
                            (r == 3) ? std::nan("") : static_cast<double>(x) / 4; break;
                case 2: {
                    char tmp[32];
                    const int len = std::snprintf(tmp, sizeof(tmp), "k%lld-%s",
                                                  static_cast<long long>(x + 500),
                                                  (r % 3) ? "long-enough-to-spill" : "s");
                    StringView sv = StringView::from_cstr(tmp);
                    if (len > 12) {
                        std::memcpy(pool + used, tmp, static_cast<size_t>(len));
                        sv.ref.buf_idx = 0;
                        sv.ref.offset = used;
                        used += static_cast<uint32_t>(len);
                    }
                    static_cast<StringView*>(col.data)[r] = sv;
                    break;
                }
                case 3: static_cast<uint32_t*>(col.data)[r] = static_cast<uint32_t>(x + 1000); break;
                case 4: std::memset(static_cast<uint8_t*>(col.data) + r * 16, int(r & 0xFF), 16); break;
                default: static_cast<uint8_t*>(col.data)[r] = static_cast<uint8_t>(r & 1); break;
            }
        }
        if (c == 2) col.str_overflow_base = pool;
        if (c == 0 || c == 2) { col.validity = v; col.stats.all_valid = false; }
        b->columns[0][c] = col;
        b->columns[1][c] = col;
    }
}

FrameMeta meta(uint64_t lsn, int64_t ts, FrameOpKind k = FrameOpKind::kBatch) {
    FrameMeta m;
    std::memset(&m, 0, sizeof(m));
    m.kind = k;
    m.flags = 0x5;
    m.lsn_min = lsn;
    m.lsn_max = lsn + 2;
    m.ts_min = ts;
    m.ts_max = ts + 10;
    m.txn_id = 77;
    m.first_pos = lsn * 10;
    std::memcpy(m.key_min, "aaaaaaaa", 8);
    std::memcpy(m.key_max, "zzzzzzzz", 8);
    return m;
}

}  // namespace

TEST(WireFrame, RoundTripZoneMapsAndZeroCopyView) {
    Arena a;
    BoltBatch src;
    build_batch(&src, &a, 300, 1);
    const size_t need = frame_size(&src, true);
    ASSERT_GT(need, 0u);
    Buf buf(need);
    FrameStatus st;
    ASSERT_EQ(frame_write(&src, meta(10, 1000), FrameZones::kCompute, nullptr, nullptr,
                          buf.p, buf.n, &st), need);
    ASSERT_EQ(st, FrameStatus::kOk);
    FrameView v;
    ASSERT_EQ(frame_parse(buf.p, buf.n, true, &v), FrameStatus::kOk);
    EXPECT_EQ(v.frame_len, need);
    EXPECT_EQ(v.header.lsn, 10u);
    EXPECT_EQ(v.trailer.lsn_max, 12u);
    EXPECT_EQ(v.trailer.ts_max, 1010);
    EXPECT_EQ(v.header.rows, 300u);
    EXPECT_EQ(v.header.txn_id, 77u);
    EXPECT_EQ(v.n_zones, 6u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(v.zones) % 32, 0u);

    // Zones vs a model over the source.
    int64_t mn = INT64_MAX, mx = INT64_MIN;
    double fmn = 1e300, fmx = -1e300;
    uint32_t nulls = 0;
    for (int64_t r = 0; r < 300; ++r) {
        const double f = static_cast<const double*>(src.col(1).data)[r];
        if (f == f) { fmn = std::min(fmn, f); fmx = std::max(fmx, f); }
        if (src.col(0).is_null(r)) { ++nulls; continue; }
        const int64_t x = static_cast<const int64_t*>(src.col(0).data)[r];
        mn = std::min(mn, x); mx = std::max(mx, x);
    }
    EXPECT_EQ(v.zone_kinds[0], uint8_t(FrameZoneKind::kI64));
    EXPECT_EQ(v.zones[0].min_value, mn);
    EXPECT_EQ(v.zones[0].max_value, mx);
    EXPECT_EQ(v.zones[0].null_count, nulls);
    EXPECT_EQ(v.zone_kinds[1], uint8_t(FrameZoneKind::kF64));
    EXPECT_EQ(zone_min_f64(&v.zones[1]), fmn);
    EXPECT_EQ(zone_max_f64(&v.zones[1]), fmx);
    EXPECT_EQ(v.zone_kinds[2], uint8_t(FrameZoneKind::kStr8));
    EXPECT_EQ(v.zone_kinds[3], uint8_t(FrameZoneKind::kU64));
    EXPECT_EQ(v.zone_kinds[4], uint8_t(FrameZoneKind::kNone));
    EXPECT_EQ(v.zone_kinds[5], uint8_t(FrameZoneKind::kI64));
    EXPECT_EQ(v.zones[5].min_value, 0);
    EXPECT_EQ(v.zones[5].max_value, 1);

    // The payload is a wire batch: zero-copy view, buffers inside the frame
    // and 64 B aligned.
    BoltBatch dst;
    ASSERT_TRUE(bolt_wire_view(v.payload, v.payload_len, &dst, &a));
    for (uint32_t c = 0; c < 6; ++c) {
        const uint8_t* d = static_cast<const uint8_t*>(dst.col(c).data);
        ASSERT_GE(d, buf.p);
        ASSERT_LT(d, buf.p + need);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(d) % 64, 0u);
    }
    for (int64_t r = 0; r < 300; ++r) {
        ASSERT_EQ(dst.col(0).is_null(r), src.col(0).is_null(r));
        if (!src.col(0).is_null(r))
            ASSERT_EQ(static_cast<const int64_t*>(dst.col(0).data)[r],
                      static_cast<const int64_t*>(src.col(0).data)[r]);
        const uint8_t *p1, *p2;
        int32_t n1, n2;
        if (src.col(2).is_null(r)) continue;
        src.col(2).utf8_at(r, &p1, &n1);
        dst.col(2).utf8_at(r, &p2, &n2);
        ASSERT_EQ(n1, n2);
        ASSERT_EQ(std::memcmp(p1, p2, static_cast<size_t>(n1)), 0);
    }
}

TEST(WireFrame, DeterministicBytesWhateverTheBufferHeld) {
    Arena a;
    BoltBatch src;
    build_batch(&src, &a, 77, 2);
    const size_t need = frame_size(&src, true);
    Buf x(need, 0x00), y(need, 0xAB);
    FrameStatus st;
    ASSERT_EQ(frame_write(&src, meta(5, 5), FrameZones::kCompute, nullptr, nullptr, x.p, x.n, &st), need);
    ASSERT_EQ(frame_write(&src, meta(5, 5), FrameZones::kCompute, nullptr, nullptr, y.p, y.n, &st), need);
    EXPECT_EQ(std::memcmp(x.p, y.p, need), 0);
}

TEST(WireFrame, EveryByteFlipIsDetected) {
    Arena a;
    BoltBatch src;
    build_batch(&src, &a, 40, 3);
    const size_t need = frame_size(&src, true);
    Buf buf(need);
    FrameStatus st;
    ASSERT_EQ(frame_write(&src, meta(1, 1), FrameZones::kCompute, nullptr, nullptr, buf.p, buf.n, &st), need);
    std::vector<uint8_t> good(buf.p, buf.p + need);
    for (size_t i = 0; i < need; ++i) {
        for (uint8_t bit : {uint8_t(0x01), uint8_t(0x80)}) {
            std::memcpy(buf.p, good.data(), need);
            buf.p[i] ^= bit;
            FrameView v;
            const FrameStatus s = frame_parse(buf.p, need, true, &v);
            ASSERT_NE(s, FrameStatus::kOk) << "flip at " << i;
        }
    }
}

TEST(WireFrame, ProvidedNoneAndRefusals) {
    Arena a;
    BoltBatch src;
    build_batch(&src, &a, 10, 4);
    Buf buf(frame_size(&src, true));
    FrameStatus st;
    // Zones off: trailer is just the 64 B fixed part.
    const size_t n0 = frame_write(&src, meta(1, 1), FrameZones::kNone, nullptr, nullptr, buf.p, buf.n, &st);
    ASSERT_EQ(n0, frame_size(&src, false));
    FrameView v;
    ASSERT_EQ(frame_parse(buf.p, n0, true, &v), FrameStatus::kOk);
    EXPECT_EQ(v.n_zones, 0u);
    // Provided zones pass through verbatim.
    ZoneMap z[6];
    uint8_t k[6];
    for (int i = 0; i < 6; ++i) { z[i] = zone_make_empty_i64(); z[i].min_value = i; z[i].max_value = 100 + i; k[i] = 1; }
    ASSERT_GT(frame_write(&src, meta(1, 1), FrameZones::kProvided, z, k, buf.p, buf.n, &st), 0u);
    ASSERT_EQ(frame_parse(buf.p, buf.n, true, &v), FrameStatus::kOk);
    EXPECT_EQ(v.zones[4].max_value, 104);
    // Refusals: too small, kNone op, inverted lsn range.
    EXPECT_EQ(frame_write(&src, meta(1, 1), FrameZones::kCompute, nullptr, nullptr, buf.p, 100, &st), 0u);
    EXPECT_EQ(st, FrameStatus::kCapacity);
    FrameMeta bad = meta(1, 1, FrameOpKind::kNone);
    EXPECT_EQ(frame_write(&src, bad, FrameZones::kNone, nullptr, nullptr, buf.p, buf.n, &st), 0u);
    bad = meta(9, 1);
    bad.lsn_max = 3;
    EXPECT_EQ(frame_write(&src, bad, FrameZones::kNone, nullptr, nullptr, buf.p, buf.n, &st), 0u);
    EXPECT_EQ(st, FrameStatus::kBadLayout);
    // A zero header is the end of a run.
    Buf zero(256);
    EXPECT_EQ(frame_parse(zero.p, zero.n, true, &v), FrameStatus::kEnd);
}

namespace {

struct FileFixture {
    std::vector<FrameIndexEntry> index = std::vector<FrameIndexEntry>(64);
    std::vector<ZoneMap> rollup = std::vector<ZoneMap>(16);
    std::vector<uint8_t> kinds = std::vector<uint8_t>(16);
};

}  // namespace

TEST(WireFrameFile, AppendSealOpenReadEveryFrame) {
    Arena a;
    Buf file(4 << 20);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kWalSegment, 7, 9, 0x5eedull,
                               fx.index.data(), 64, fx.rollup.data(), fx.kinds.data(), 16),
              FrameStatus::kOk);
    int64_t mn = INT64_MAX, mx = INT64_MIN;
    uint64_t rows = 0;
    for (int f = 0; f < 9; ++f) {
        BoltBatch b;
        build_batch(&b, &a, 13 + f * 37, f);
        for (int64_t r = 0; r < b.num_rows; ++r) {
            if (b.col(0).is_null(r)) continue;
            const int64_t x = static_cast<const int64_t*>(b.col(0).data)[r];
            mn = std::min(mn, x); mx = std::max(mx, x);
        }
        rows += static_cast<uint64_t>(b.num_rows);
        const FrameOpKind k = f == 4 ? FrameOpKind::kDeleteKeys : FrameOpKind::kBatch;
        ASSERT_EQ(frame_file_append(&w, &b, meta(100 + 3u * uint64_t(f), 50 + f, k), FrameZones::kCompute,
                                    nullptr, nullptr), FrameStatus::kOk);
    }
    const size_t len = frame_file_seal(&w);
    ASSERT_GT(len, 0u);
    EXPECT_EQ(len % kFrameChunkAlign, 0u);

    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    ASSERT_TRUE(v.sealed);
    EXPECT_EQ(v.n_frames, 9u);
    EXPECT_EQ(v.footer.total_rows, rows);
    EXPECT_EQ(v.footer.lsn_min, 100u);
    EXPECT_EQ(v.footer.lsn_max, 100u + 24 + 2);
    EXPECT_EQ(v.header.id0, 7u);
    EXPECT_EQ(v.header.purpose, uint16_t(FrameFilePurpose::kWalSegment));
    EXPECT_EQ(v.footer.index_off % kFrameChunkAlign, 0u);
    ASSERT_EQ(v.n_rollup, 6u);
    EXPECT_EQ(v.rollup[0].min_value, mn);
    EXPECT_EQ(v.rollup[0].max_value, mx);
    EXPECT_EQ(v.rollup_kind[4], uint8_t(FrameZoneKind::kNone));
    for (uint32_t i = 0; i < v.n_frames; ++i) {
        FrameView fv;
        ASSERT_EQ(frame_file_frame(v, i, true, &fv), FrameStatus::kOk);
        EXPECT_EQ(v.index[i].off % 64, 0u);
        EXPECT_EQ(fv.header.kind, uint16_t(i == 4 ? FrameOpKind::kDeleteKeys : FrameOpKind::kBatch));
        BoltBatch d;
        ASSERT_TRUE(bolt_wire_view(fv.payload, fv.payload_len, &d, &a));
        EXPECT_EQ(d.num_rows, 13 + int64_t(i) * 37);
    }
    // The recovery walk of a sealed file sees the same frames.
    std::vector<FrameIndexEntry> rec(64);
    size_t end = 0;
    FrameStatus why;
    ASSERT_EQ(frame_file_recover(v, rec.data(), 64, &end, &why), 9);
    EXPECT_EQ(std::memcmp(rec.data(), v.index, 9 * sizeof(FrameIndexEntry)), 0);
}

TEST(WireFrameFile, RollupDroppedWhenSchemaChanges) {
    Arena a;
    Buf file(1 << 20);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kSpill, 0, 0, 0x5eedull,
                               fx.index.data(), 64, fx.rollup.data(), fx.kinds.data(), 16),
              FrameStatus::kOk);
    BoltBatch b;
    build_batch(&b, &a, 20, 0);
    ASSERT_EQ(frame_file_append(&w, &b, meta(1, 1), FrameZones::kCompute, nullptr, nullptr), FrameStatus::kOk);
    b.num_cols = 5;   // a narrower batch: column count differs
    b.schema.num_fields = 5;
    ASSERT_EQ(frame_file_append(&w, &b, meta(4, 2), FrameZones::kCompute, nullptr, nullptr), FrameStatus::kOk);
    const size_t len = frame_file_seal(&w);
    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    EXPECT_EQ(v.n_frames, 2u);
    EXPECT_EQ(v.n_rollup, 0u);
}

TEST(WireFrameFile, CrashBeforeSealRecoverResumeSeal) {
    Arena a;
    Buf file(2 << 20);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kWalSegment, 1, 2, 0x5eedull,
                               fx.index.data(), 64, fx.rollup.data(), fx.kinds.data(), 16),
              FrameStatus::kOk);
    for (int f = 0; f < 5; ++f) {
        BoltBatch b;
        build_batch(&b, &a, 50 + f, f);
        ASSERT_EQ(frame_file_append(&w, &b, meta(10u * uint64_t(f) + 1, f), FrameZones::kCompute, nullptr, nullptr),
                  FrameStatus::kOk);
    }
    // Crash: no seal, and frame 3 is torn (a byte of its payload is wrong).
    file.p[fx.index[3].off + 200] ^= 0x10;

    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, file.n, &v), FrameStatus::kOk);
    EXPECT_FALSE(v.sealed);
    std::vector<FrameIndexEntry> rec(64);
    size_t end = 0;
    FrameStatus why;
    ASSERT_EQ(frame_file_recover(v, rec.data(), 64, &end, &why), 3);
    EXPECT_EQ(why, FrameStatus::kBadCrc);
    EXPECT_EQ(end, fx.index[3].off);

    FileFixture fx2;
    FrameFileWriter w2;
    ASSERT_EQ(frame_file_resume(&w2, file.p, file.n, fx2.index.data(), 64, fx2.rollup.data(),
                                fx2.kinds.data(), 16), FrameStatus::kOk);
    EXPECT_EQ(w2.n_frames, 3u);
    EXPECT_EQ(w2.pos, end);
    for (int f = 0; f < 2; ++f) {
        BoltBatch b;
        build_batch(&b, &a, 9, 40 + f);   // shorter than the torn frame
        ASSERT_EQ(frame_file_append(&w2, &b, meta(100u + uint64_t(f), 9), FrameZones::kCompute, nullptr, nullptr),
                  FrameStatus::kOk);
    }
    const size_t len = frame_file_seal(&w2);
    ASSERT_GT(len, 0u);
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    ASSERT_TRUE(v.sealed);
    ASSERT_EQ(v.n_frames, 5u);
    EXPECT_EQ(v.index[3].lsn_min, 100u);
    for (uint32_t i = 0; i < 5; ++i) {
        FrameView fv;
        ASSERT_EQ(frame_file_frame(v, i, true, &fv), FrameStatus::kOk);
    }
}

TEST(WireFrameFile, HeaderUnitsValidatedAndNamed) {
    Buf file(1 << 16);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kGeneric, 0, 0, 0x5eedull,
                               fx.index.data(), 64, nullptr, nullptr, 0), FrameStatus::kOk);
    const size_t len = frame_file_seal(&w);
    ASSERT_GT(len, 0u);
    auto patch = [&](size_t off, uint8_t val) {
        FrameFileHeader h;
        std::memcpy(&h, file.p, sizeof(h));
        reinterpret_cast<uint8_t*>(&h)[off] = val;
        h.header_crc32c = io::crc32c(&h, 60);
        std::memcpy(file.p, &h, sizeof(h));
    };
    FrameFileView v;
    const FrameFileHeader good = w.header;
    struct Case { size_t off; uint8_t val; FrameStatus want; } cases[] = {
        {8, 5, FrameStatus::kBadBufAlign},      // 32 B
        {8, 13, FrameStatus::kBadBufAlign},     // 8 KiB
        {9, 11, FrameStatus::kBadChunkAlign},   // 2 KiB
        {9, 21, FrameStatus::kBadChunkAlign},   // 2 MiB
        {10, 13, FrameStatus::kBadStripeAlign}, // 8 KiB
        {10, 24, FrameStatus::kBadStripeAlign}, // 16 MiB
        {11, 8, FrameStatus::kBadIoAlign},      // 256 B
        {4, 2, FrameStatus::kBadVersion},
    };
    for (const Case& c : cases) {
        std::memcpy(file.p, &good, sizeof(good));
        patch(c.off, c.val);
        EXPECT_EQ(frame_file_open(file.p, len, &v), c.want) << c.off << "=" << int(c.val);
    }
    // chunk > stripe is refused as a stripe error.
    std::memcpy(file.p, &good, sizeof(good));
    patch(9, 18);   // 256 KiB chunk, 64 KiB stripe
    EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kBadStripeAlign);
    // A stale CRC and a wrong magic.
    std::memcpy(file.p, &good, sizeof(good));
    file.p[16] ^= 1;
    EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kBadCrc);
    std::memcpy(file.p, &good, sizeof(good));
    file.p[0] = 'X';
    EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kBadMagic);
    std::memcpy(file.p, &good, sizeof(good));
    EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
}

// Files written at chunk 4 / 16 / 64 KiB (and a 16 KiB stripe) all read the
// same rows; only the mappability of a chunk region differs per host.
TEST(WireFrameFile, AlignmentMatrixReadsIdentically) {
    Arena a;
    const FrameAlign units[4] = {{6, 12, 16, 12}, {6, 14, 16, 14}, {6, 16, 16, 12}, {6, 14, 14, 12}};
    std::vector<int64_t> first_sum;
    for (const FrameAlign& u : units) {
        Buf file(1 << 20);
        FileFixture fx;
        FrameFileWriter w;
        ASSERT_EQ(frame_file_begin_with(&w, file.p, file.n, u, FrameFilePurpose::kL0Run, 0, 0, 0x5eedull,
                                        fx.index.data(), 64, fx.rollup.data(), fx.kinds.data(), 16),
                  FrameStatus::kOk);
        for (int f = 0; f < 3; ++f) {
            BoltBatch b;
            build_batch(&b, &a, 30, f);
            ASSERT_EQ(frame_file_append(&w, &b, meta(uint64_t(f) + 1, f), FrameZones::kCompute, nullptr, nullptr),
                      FrameStatus::kOk);
        }
        const size_t len = frame_file_seal(&w);
        ASSERT_EQ(len % (1u << u.chunk_log2), 0u);
        FrameFileView v;
        ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
        EXPECT_EQ(v.header.chunk_align_log2, u.chunk_log2);
        EXPECT_EQ(v.footer.index_off % (1u << u.chunk_log2), 0u);
        int64_t sum = 0;
        for (uint32_t i = 0; i < v.n_frames; ++i) {
            FrameView fv;
            ASSERT_EQ(frame_file_frame(v, i, true, &fv), FrameStatus::kOk);
            BoltBatch d;
            ASSERT_TRUE(bolt_wire_view(fv.payload, fv.payload_len, &d, &a));
            for (int64_t r = 0; r < d.num_rows; ++r)
                sum += static_cast<const int64_t*>(d.col(3).data)[r];
        }
        first_sum.push_back(sum);
        EXPECT_EQ(frame_file_unit_mappable(u.chunk_log2, 16384), u.chunk_log2 >= 14);
        EXPECT_TRUE(frame_file_unit_mappable(u.chunk_log2, 4096));
    }
    for (int64_t s : first_sum) EXPECT_EQ(s, first_sum[0]);
    // Writing with a non-64 B buffer unit, or invalid units, is refused.
    Buf file(1 << 16);
    FileFixture fx;
    FrameFileWriter w;
    EXPECT_EQ(frame_file_begin_with(&w, file.p, file.n, FrameAlign{7, 14, 16, 12},
                                    FrameFilePurpose::kGeneric, 0, 0, 0x5eedull, fx.index.data(), 64,
                                    nullptr, nullptr, 0), FrameStatus::kBadBufAlign);
    EXPECT_EQ(frame_file_begin_with(&w, file.p, file.n, FrameAlign{6, 17, 16, 12},
                                    FrameFilePurpose::kGeneric, 0, 0, 0x5eedull, fx.index.data(), 64,
                                    nullptr, nullptr, 0), FrameStatus::kBadStripeAlign);
}

TEST(WireFrameFile, IndexCorruptionAndCapacity) {
    Arena a;
    Buf file(1 << 20);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kShuffle, 0, 0, 0x5eedull,
                               fx.index.data(), 2, nullptr, nullptr, 0), FrameStatus::kOk);
    BoltBatch b;
    build_batch(&b, &a, 8, 0);
    ASSERT_EQ(frame_file_append(&w, &b, meta(1, 1), FrameZones::kNone, nullptr, nullptr), FrameStatus::kOk);
    ASSERT_EQ(frame_file_append(&w, &b, meta(2, 1), FrameZones::kNone, nullptr, nullptr), FrameStatus::kOk);
    EXPECT_EQ(frame_file_append(&w, &b, meta(3, 1), FrameZones::kNone, nullptr, nullptr), FrameStatus::kCapacity);
    const size_t len = frame_file_seal(&w);
    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    file.p[v.footer.index_off + 3] ^= 0x40;   // corrupt an index entry
    EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kBadCrc);
}

TEST(WireFrameFile, LimitsAreRegistered) {
    EXPECT_EQ(bolt_limits_value(bolt_limits_id::wire_buf_align), kFrameBufAlign);
    EXPECT_EQ(bolt_limits_value(bolt_limits_id::wire_chunk_align), kFrameChunkAlign);
    EXPECT_EQ(bolt_limits_value(bolt_limits_id::wire_stripe_align), kFrameStripeAlign);
    EXPECT_EQ(bolt_limits_value(bolt_limits_id::wire_io_align), kFrameIoAlign);
    const FrameAlign d = frame_align_default();
    EXPECT_EQ(frame_align_validate(d), FrameStatus::kOk);
    EXPECT_EQ(1u << d.chunk_log2, 16384u);
}

// ---------------------------------------------------------------------------
// Review additions (2026-10-02): truncation at every byte, random-schema
// container fuzz, recycled-file stale frames, recorded buf unit, hostile
// offsets, NaN zones, unaligned buffers.
// ---------------------------------------------------------------------------

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s >> 11; }
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }
};

// A random batch: 1..8 columns of fixed-width, Flat Utf8 (inline + spilled)
// and VarBinary Utf8, random nulls and NaNs, 0..max_rows rows.
void random_batch(BoltBatch* b, Arena* a, Rng* r, int64_t max_rows) {
    static const BoltType kTypes[] = {
        BoltType::Int8, BoltType::Int16, BoltType::Int32, BoltType::Int64, BoltType::UInt8,
        BoltType::UInt16, BoltType::UInt32, BoltType::UInt64, BoltType::Float32,
        BoltType::Float64, BoltType::Bool, BoltType::Date32, BoltType::Timestamp,
        BoltType::Decimal64, BoltType::Decimal128, BoltType::Utf8, BoltType::Utf8};
    const uint32_t nc = 1 + static_cast<uint32_t>(r->below(8));
    const int64_t n = static_cast<int64_t>(r->below(static_cast<uint64_t>(max_rows) + 1));
    BoltBatch::init_empty(b);
    ASSERT_TRUE(BoltBatch::alloc_columns(b, a, nc));
    b->num_rows = n;
    b->schema.num_fields = nc;
    for (uint32_t c = 0; c < nc; ++c) {
        const size_t ti = r->below(sizeof(kTypes) / sizeof(kTypes[0]));
        const BoltType t = kTypes[ti];
        char name[16];
        std::snprintf(name, sizeof(name), "c%u", c);
        set_field(b, c, name, t);
        const bool varbin = (t == BoltType::Utf8) && (ti == 16);
        BoltColumn col;
        if (t == BoltType::Utf8 && !varbin) {
            col = BoltColumn::make_flat_alloc(n, t, a);
            char* pool = static_cast<char*>(a->allocate(static_cast<size_t>(n) * 40 + 1, 64));
            uint32_t used = 0;
            for (int64_t i = 0; i < n; ++i) {
                char tmp[40];
                const int len = static_cast<int>(r->below(30));
                for (int k = 0; k < len; ++k) tmp[k] = static_cast<char>('a' + r->below(26));
                tmp[len] = '\0';
                StringView sv = StringView::from_cstr(tmp);
                if (len > 12) {
                    std::memcpy(pool + used, tmp, static_cast<size_t>(len));
                    sv.ref.buf_idx = 0;
                    sv.ref.offset = used;
                    used += static_cast<uint32_t>(len);
                }
                static_cast<StringView*>(col.data)[i] = sv;
            }
            col.str_overflow_base = pool;
        } else if (varbin) {
            int32_t* offs = static_cast<int32_t*>(a->allocate(sizeof(int32_t) * (n + 1), 64));
            uint8_t* pay = static_cast<uint8_t*>(a->allocate(static_cast<size_t>(n) * 20 + 1, 64));
            offs[0] = 0;
            for (int64_t i = 0; i < n; ++i) {
                const int len = static_cast<int>(r->below(20));
                for (int k = 0; k < len; ++k) pay[offs[i] + k] = static_cast<uint8_t>(r->next());
                offs[i + 1] = offs[i] + len;
            }
            col = BoltColumn::make_var_binary(pay, nullptr, offs, n, t, a);
        } else {
            col = BoltColumn::make_flat_alloc(n, t, a);
            uint8_t* d = static_cast<uint8_t*>(col.data);
            for (size_t k = 0; k < static_cast<size_t>(n) * col.type_size_bytes; ++k)
                d[k] = static_cast<uint8_t>(r->next());
            if (t == BoltType::Bool)
                for (int64_t i = 0; i < n; ++i) d[i] &= 1u;
            if (t == BoltType::Float64 && n > 0 && r->below(3) == 0)
                static_cast<double*>(col.data)[r->below(static_cast<uint64_t>(n))] = std::nan("");
        }
        if (n > 0 && r->below(2) == 0) {
            const size_t vb = static_cast<size_t>((n + 7) / 8);
            uint8_t* v = static_cast<uint8_t*>(a->allocate(vb, 64));
            for (size_t k = 0; k < vb; ++k) v[k] = static_cast<uint8_t>(r->next() | r->next());
            col.validity = v;
            col.stats.all_valid = false;
        }
        b->columns[0][c] = col;
        b->columns[1][c] = col;
    }
}

FrameMeta rmeta(Rng* r, uint64_t lsn) {
    FrameMeta m = meta(lsn, static_cast<int64_t>(lsn) * 3);
    m.kind = static_cast<FrameOpKind>(1 + r->below(4));
    m.flags = static_cast<uint16_t>(r->next());
    return m;
}

}  // namespace

// Every frame of a random-schema container reads back as the bytes
// bolt_wire_serialize writes for its source batch, with zones equal to a
// recomputation; the sealed index equals the walk of the same file, and the
// walk of the unsealed file equals the writer's index. Starts and padding
// are checked on every frame. BOLT_FRAME_FUZZ_FRAMES scales the run.
TEST(WireFrameFile, RandomSchemaContainerFuzz) {
    const char* env = std::getenv("BOLT_FRAME_FUZZ_FRAMES");
    const uint64_t target = env ? std::strtoull(env, nullptr, 10) : 1200u;
    uint64_t frames_done = 0;
    for (uint64_t seed = 1; frames_done < target; ++seed) {
        Rng r{seed * 0x9E3779B97F4A7C15ull};
        Arena a;
        const uint32_t nframes = 1 + static_cast<uint32_t>(r.below(64));
        Buf file(8 << 20);
        std::vector<FrameIndexEntry> idx(64);
        std::vector<ZoneMap> roll(16);
        std::vector<uint8_t> rk(16);
        FrameFileWriter w;
        ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kL0Run, seed, 0,
                                   seed ^ 0xABCDull, idx.data(), 64, roll.data(), rk.data(), 16),
                  FrameStatus::kOk);
        std::vector<std::vector<uint8_t>> wire(nframes);
        std::vector<BoltBatch> src(nframes);
        for (uint32_t f = 0; f < nframes; ++f) {
            random_batch(&src[f], &a, &r, 200);
            wire[f].resize(bolt_wire_size(&src[f]));
            ASSERT_EQ(bolt_wire_serialize(&src[f], wire[f].data(), wire[f].size()), wire[f].size());
            ASSERT_EQ(frame_file_append(&w, &src[f], rmeta(&r, 10 + 3ull * f),
                                        r.below(4) ? FrameZones::kCompute : FrameZones::kNone,
                                        nullptr, nullptr), FrameStatus::kOk) << "seed " << seed;
        }
        // Footer absent: the walk equals the writer's own index.
        FrameFileView uv;
        ASSERT_EQ(frame_file_open(file.p, file.n, &uv), FrameStatus::kOk);
        ASSERT_FALSE(uv.sealed);
        std::vector<FrameIndexEntry> walk(64);
        size_t end = 0;
        FrameStatus why;
        ASSERT_EQ(frame_file_recover(uv, walk.data(), 64, &end, &why), int64_t(nframes));
        EXPECT_EQ(why, FrameStatus::kEnd);
        EXPECT_EQ(std::memcmp(walk.data(), idx.data(), nframes * sizeof(FrameIndexEntry)), 0);
        const size_t len = frame_file_seal(&w);
        ASSERT_GT(len, 0u);
        FrameFileView v;
        ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
        ASSERT_TRUE(v.sealed);
        ASSERT_EQ(v.n_frames, nframes);
        ASSERT_EQ(frame_file_recover(v, walk.data(), 64, &end, &why), int64_t(nframes));
        EXPECT_EQ(std::memcmp(walk.data(), v.index, nframes * sizeof(FrameIndexEntry)), 0);
        for (uint32_t f = 0; f < nframes; ++f) {
            FrameView fv;
            ASSERT_EQ(frame_file_frame(v, f, true, &fv), FrameStatus::kOk);
            EXPECT_EQ(v.index[f].off % 64, 0u);
            ASSERT_EQ(fv.payload_len, wire[f].size());
            ASSERT_EQ(std::memcmp(fv.payload, wire[f].data(), wire[f].size()), 0);
            const size_t pad_end = sizeof(FrameHeader) + ((fv.payload_len + 63) & ~size_t(63));
            for (size_t k = sizeof(FrameHeader) + fv.payload_len; k < pad_end; ++k)
                ASSERT_EQ(file.p[v.index[f].off + k], 0u);
            BoltBatch d;
            ASSERT_TRUE(bolt_wire_view(fv.payload, fv.payload_len, &d, &a));
            ASSERT_EQ(d.num_rows, src[f].num_rows);
            for (uint32_t z = 0; z < fv.n_zones; ++z) {
                ZoneMap want;
                const FrameZoneKind k = frame_zone_for_column(d.col(z), &want);
                ASSERT_EQ(fv.zone_kinds[z], uint8_t(k));
                ASSERT_EQ(std::memcmp(&fv.zones[z], &want, sizeof(ZoneMap)), 0)
                    << "seed " << seed << " frame " << f << " col " << z;
            }
        }
        frames_done += nframes;
    }
}

// Cut the file at every byte of the last frame (absent tail, and a garbage
// tail): recovery keeps exactly the whole frames before it, resume + seal
// produce a file that reads back.
TEST(WireFrameFile, TruncationAtEveryByteOfTheLastFrame) {
    Arena a;
    Buf file(1 << 18);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kWalSegment, 1, 1, 42,
                               fx.index.data(), 64, fx.rollup.data(), fx.kinds.data(), 16),
              FrameStatus::kOk);
    for (int f = 0; f < 4; ++f) {
        BoltBatch b;
        build_batch(&b, &a, 6 + f, f);
        ASSERT_EQ(frame_file_append(&w, &b, meta(uint64_t(f) + 1, f), FrameZones::kCompute,
                                    nullptr, nullptr), FrameStatus::kOk);
    }
    const FrameIndexEntry last = fx.index[3];
    const size_t full = static_cast<size_t>(last.off + last.len);
    std::vector<uint8_t> good(file.p, file.p + full + 64);
    uint64_t cuts = 0;
    for (size_t t = static_cast<size_t>(last.off); t < full; ++t) {
        for (int garbage = 0; garbage < 2; ++garbage) {
            Buf cut(full + 40000);
            std::memcpy(cut.p, good.data(), t);
            std::memset(cut.p + t, garbage ? 0xA5 : 0x00, cut.n - t);
            const size_t len = garbage ? cut.n : t;
            FrameFileView v;
            ASSERT_EQ(frame_file_open(cut.p, len, &v), FrameStatus::kOk) << t;
            std::vector<FrameIndexEntry> rec(8);
            size_t end = 0;
            FrameStatus why;
            ASSERT_EQ(frame_file_recover(v, rec.data(), 8, &end, &why), 3) << "cut " << t;
            ASSERT_EQ(end, static_cast<size_t>(last.off));
            ASSERT_EQ(std::memcmp(rec.data(), fx.index.data(), 3 * sizeof(FrameIndexEntry)), 0);
            FileFixture fx2;
            FrameFileWriter w2;
            ASSERT_EQ(frame_file_resume(&w2, cut.p, len, fx2.index.data(), 64,
                                        fx2.rollup.data(), fx2.kinds.data(), 16), FrameStatus::kOk);
            ASSERT_EQ(w2.n_frames, 3u) << "cut " << t << " garbage " << garbage;
            if (garbage && t % 97 == 0) {   // reseal a sample: the file reads back whole
                const size_t sl = frame_file_seal(&w2);
                ASSERT_GT(sl, 0u);
                FrameFileView sv;
                ASSERT_EQ(frame_file_open(cut.p, sl, &sv), FrameStatus::kOk);
                ASSERT_TRUE(sv.sealed);
                ASSERT_EQ(sv.n_frames, 3u);
            }
            ++cuts;
        }
    }
    EXPECT_EQ(cuts, 2 * last.len);
}

// A recycled, non-zeroed buffer: file B is begun over sealed file A. B's
// frames are byte-for-byte the same batches as A's, so an unseeded CRC would
// accept A's old frames and A's old footer as B's. The seeded CRC refuses
// both even when B's terminating zero slot was lost.
TEST(WireFrameFile, RecycledBufferStaleFramesAndFooterRefused) {
    Arena a;
    Buf file(1 << 18);
    BoltBatch batches[5];
    for (int f = 0; f < 5; ++f) build_batch(&batches[f], &a, 20, f);
    FileFixture fa;
    FrameFileWriter wa;
    ASSERT_EQ(frame_file_begin(&wa, file.p, file.n, FrameFilePurpose::kWalSegment, 3, 7, 1,
                               fa.index.data(), 64, nullptr, nullptr, 0), FrameStatus::kOk);
    for (int f = 0; f < 5; ++f)
        ASSERT_EQ(frame_file_append(&wa, &batches[f], meta(uint64_t(f) + 1, 1), FrameZones::kNone,
                                    nullptr, nullptr), FrameStatus::kOk);
    const size_t alen = frame_file_seal(&wa);
    ASSERT_GT(alen, 0u);
    std::vector<uint8_t> a_bytes(file.p, file.p + alen);

    // Same ids, same batches, new incarnation; two frames, then a crash in
    // which the zeroing of the next header slot did not reach the media.
    FileFixture fb;
    FrameFileWriter wb;
    ASSERT_EQ(frame_file_begin(&wb, file.p, file.n, FrameFilePurpose::kWalSegment, 3, 7, 2,
                               fb.index.data(), 64, nullptr, nullptr, 0), FrameStatus::kOk);
    for (int f = 0; f < 2; ++f)
        ASSERT_EQ(frame_file_append(&wb, &batches[f], meta(uint64_t(f) + 1, 1), FrameZones::kNone,
                                    nullptr, nullptr), FrameStatus::kOk);
    ASSERT_EQ(wb.pos, fa.index[2].off);
    std::memcpy(file.p + wb.pos, a_bytes.data() + wb.pos, sizeof(FrameHeader));

    FrameFileView v;
    EXPECT_EQ(frame_file_open(file.p, alen, &v), FrameStatus::kBadCrc);   // A's footer
    ASSERT_NE(v.base, nullptr);
    std::vector<FrameIndexEntry> rec(64);
    size_t end = 0;
    FrameStatus why;
    EXPECT_EQ(frame_file_recover(v, rec.data(), 64, &end, &why), 2);
    EXPECT_EQ(why, FrameStatus::kBadCrc);
    FileFixture fr;
    FrameFileWriter wr;
    ASSERT_EQ(frame_file_resume(&wr, file.p, alen, fr.index.data(), 64, nullptr, nullptr, 0),
              FrameStatus::kOk);
    EXPECT_EQ(wr.n_frames, 2u);
    // A genuinely sealed file is still refused by resume.
    std::memcpy(file.p, a_bytes.data(), alen);
    EXPECT_EQ(frame_file_resume(&wr, file.p, alen, fr.index.data(), 64, nullptr, nullptr, 0),
              FrameStatus::kBadLayout);
}

// A reader honours a recorded buf unit larger than 64 B: frames on 128 B
// boundaries with zero gaps are all found; the 64 B writer refuses to resume.
TEST(WireFrameFile, ReaderHonoursRecordedBufUnit) {
    Arena a;
    Buf file(1 << 16);
    FrameFileHeader h;
    std::memset(&h, 0, sizeof(h));
    std::memcpy(h.magic, "BWFF", 4);
    h.version = kFrameFileVersion;
    h.buf_align_log2 = 7;
    h.chunk_align_log2 = 14;
    h.stripe_align_log2 = 16;
    h.io_align_log2 = 12;
    h.incarnation = 9;
    h.frames_off = 128;
    h.header_crc32c = io::crc32c(&h, 60);
    std::memcpy(file.p, &h, sizeof(h));
    size_t off = 128;
    for (int f = 0; f < 3; ++f) {
        BoltBatch b;
        build_batch(&b, &a, 5 + f, f);
        FrameMeta m = meta(uint64_t(f) + 1, 1);
        m.crc_seed = h.header_crc32c;
        FrameStatus st;
        const size_t n = frame_write(&b, m, FrameZones::kNone, nullptr, nullptr, file.p + off,
                                     file.n - off, &st);
        ASSERT_GT(n, 0u);
        off += (n + 127) & ~size_t(127);
    }
    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, file.n, &v), FrameStatus::kOk);
    std::vector<FrameIndexEntry> rec(8);
    size_t end = 0;
    FrameStatus why;
    EXPECT_EQ(frame_file_recover(v, rec.data(), 8, &end, &why), 3);
    EXPECT_EQ(end, off);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(rec[i].off % 128, 0u);
    FileFixture fx;
    FrameFileWriter w;
    EXPECT_EQ(frame_file_resume(&w, file.p, file.n, fx.index.data(), 64, nullptr, nullptr, 0),
              FrameStatus::kBadBufAlign);
}

// Offsets near 2^64 in a footer or an index entry (CRCs recomputed, as a
// hostile writer would) are refused, never wrapped into a read.
TEST(WireFrameFile, HostileOffsetsRefusedWithoutOverflow) {
    Arena a;
    Buf file(1 << 18);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kGeneric, 0, 0, 5,
                               fx.index.data(), 64, nullptr, nullptr, 0), FrameStatus::kOk);
    BoltBatch b;
    build_batch(&b, &a, 10, 1);
    ASSERT_EQ(frame_file_append(&w, &b, meta(1, 1), FrameZones::kNone, nullptr, nullptr), FrameStatus::kOk);
    const size_t len = frame_file_seal(&w);
    FrameFileView v;
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    const uint32_t seed = v.seed;
    FrameFileFooter f = v.footer;
    const FrameFileFooter good = f;
    for (uint64_t bad_off : {~0ull - 16383ull, ~0ull - 65535ull, uint64_t(1) << 63}) {
        f = good;
        f.index_off = bad_off;
        f.footer_crc32c = io::crc32c(&f, 60, seed);
        std::memcpy(file.p + len - 64, &f, 64);
        EXPECT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kBadLayout) << bad_off;
    }
    std::memcpy(file.p + len - 64, &good, 64);
    FrameIndexEntry e;
    std::memcpy(&e, file.p + good.index_off, sizeof(e));
    e.off = ~0ull - 63ull;
    std::memcpy(file.p + good.index_off, &e, sizeof(e));
    f = good;
    f.index_crc32c = io::crc32c(file.p + good.index_off, good.index_len, seed);
    f.footer_crc32c = io::crc32c(&f, 60, seed);
    std::memcpy(file.p + len - 64, &f, 64);
    ASSERT_EQ(frame_file_open(file.p, len, &v), FrameStatus::kOk);
    FrameView fv;
    EXPECT_EQ(frame_file_frame(v, 0, true, &fv), FrameStatus::kBadLayout);
}

TEST(WireFrame, NaNNeverSeedsAZone) {
    Arena a;
    BoltColumn c = BoltColumn::make_flat_alloc(4, BoltType::Float64, &a);
    double* d = static_cast<double*>(c.data);
    d[0] = std::nan(""); d[1] = 5.0; d[2] = 7.5; d[3] = std::nan("");
    ZoneMap z;
    ASSERT_EQ(frame_zone_for_column(c, &z), FrameZoneKind::kF64);
    EXPECT_EQ(zone_min_f64(&z), 5.0);
    EXPECT_EQ(zone_max_f64(&z), 7.5);
    d[1] = std::nan(""); d[2] = std::nan("");
    EXPECT_EQ(frame_zone_for_column(c, &z), FrameZoneKind::kNone);   // no bound, not [0, 0]
}

TEST(WireFrame, UnalignedReaderBufferRefused) {
    Arena a;
    BoltBatch src;
    build_batch(&src, &a, 9, 1);
    const size_t need = frame_size(&src, true);
    Buf buf(need + 64);
    FrameStatus st;
    EXPECT_EQ(frame_write(&src, meta(1, 1), FrameZones::kCompute, nullptr, nullptr, buf.p + 8,
                          need, &st), 0u);
    EXPECT_EQ(st, FrameStatus::kUnaligned);
    ASSERT_EQ(frame_write(&src, meta(1, 1), FrameZones::kCompute, nullptr, nullptr, buf.p,
                          need, &st), need);
    std::memmove(buf.p + 8, buf.p, need);
    FrameView v;
    EXPECT_EQ(frame_parse(buf.p + 8, need, true, &v), FrameStatus::kUnaligned);
    FrameFileView fv;
    EXPECT_EQ(frame_file_open(buf.p + 8, need, &fv), FrameStatus::kUnaligned);
}
