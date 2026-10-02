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
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kWalSegment, 7, 9,
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
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kSpill, 0, 0,
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
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kWalSegment, 1, 2,
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
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kGeneric, 0, 0,
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
        ASSERT_EQ(frame_file_begin_with(&w, file.p, file.n, u, FrameFilePurpose::kL0Run, 0, 0,
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
                                    FrameFilePurpose::kGeneric, 0, 0, fx.index.data(), 64,
                                    nullptr, nullptr, 0), FrameStatus::kBadBufAlign);
    EXPECT_EQ(frame_file_begin_with(&w, file.p, file.n, FrameAlign{6, 17, 16, 12},
                                    FrameFilePurpose::kGeneric, 0, 0, fx.index.data(), 64,
                                    nullptr, nullptr, 0), FrameStatus::kBadStripeAlign);
}

TEST(WireFrameFile, IndexCorruptionAndCapacity) {
    Arena a;
    Buf file(1 << 20);
    FileFixture fx;
    FrameFileWriter w;
    ASSERT_EQ(frame_file_begin(&w, file.p, file.n, FrameFilePurpose::kShuffle, 0, 0,
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
