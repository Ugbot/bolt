#include <gtest/gtest.h>
#include "bolt/wire/xchannel.h"
#include <array>
#include <thread>

namespace {
namespace xc = bolt::wire::xchannel;
using Status = xc::Status;
// Independently literal FORMAT4 prefix: not made by the parser's constants or
// a paired writer. Identity CRC FB04EE98 is upstream v4.rs's initial-file golden.
constexpr std::array<std::uint8_t, 208> kInitial{
    1,0,0,0,192,0,0,0, 0,0,0,0,0,0,0,0,
    224,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,16,0,0,0,4,0,0, 4,0,1,8,0,0,0,0,
    8,'X','C',0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0, 68,0,0,0,0,0,0,0,
    1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
    255,255,255,255,255,255,255,255, 0,0,0,0,0,0,0,0,
    1,0,0,0,152,238,4,251, 0,0,0,0,0,0,0,0
};
constexpr std::array<std::uint8_t, 24> kUser{
    1,1,0x34,0x12,3,0,0,0, 8,7,6,5,4,3,2,1,
    0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x99,0x88
};
// Upstream's independently published Roll golden: seq8, id21..30, index1000.
constexpr std::array<std::uint8_t, 56> kRoll{
    1,3,0,0,40,0,0,0, 0,0,0,0,0,0,0,0,
    8,0,0,0,0,0,0,0, 0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x2b,0x2c,0x2d,0x2e,0x2f,0x30, 0xe8,3,0,0,0,0,0,0,
    0x0b,0x10,0xd7,0xf9,0,0,0,0
};
constexpr xc::Limits kLimits{4096, 8192, 2048};
constexpr xc::Expected kExpected{0, 68, 0, true};
template <typename T> void put(std::uint8_t* p, T value) {
    for (std::uint32_t i = 0; i < sizeof(T); ++i) p[i] = std::uint8_t(value >> (i * 8));
}
// Test-only reference CRC for mutations, independent bit recurrence rather
// than production Bolt CRC. Golden acceptance tests use literal CRCs above.
std::uint32_t reference_crc(const std::uint8_t* p, std::uint32_t n) {
    std::uint32_t c = UINT32_MAX;
    for (std::uint32_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (std::uint32_t bit = 0; bit < 8; ++bit)
            c = (c >> 1) ^ ((c & 1) != 0 ? 0x82f63b78u : 0u);
    }
    return ~c;
}
struct alignas(8) Fixture {
    std::array<std::uint8_t, 8192> bytes{};
    Fixture() { std::memcpy(bytes.data(), kInitial.data(), kInitial.size()); }
    xc::Region region(std::uint32_t n = 4096, std::uint64_t off = 0) {
        return {{bytes.data() + off, n}, off};
    }
    xc::File file(xc::Expected e = kExpected, xc::Limits l = kLimits) {
        xc::File f{};
        EXPECT_EQ(xc::admit_file({bytes.data(), 208}, e, l, f), Status::Ok);
        return f;
    }
    void identity_crc() { put(bytes.data() + 196, reference_crc(bytes.data() + 144, 40)); }
};
void seed_roll(Fixture& old, Fixture& next) {
    put<std::uint64_t>(old.bytes.data() + 24, 100);
    put<std::uint64_t>(old.bytes.data() + 32, 900);
    put<std::uint64_t>(old.bytes.data() + 40, 7);
    std::memcpy(old.bytes.data() + 208, kRoll.data(), 56);
    put<std::uint64_t>(next.bytes.data() + 32, 1000);
    put<std::uint64_t>(next.bytes.data() + 40, 8);
    for (std::uint32_t i = 0; i < 16; ++i) next.bytes[144 + i] = std::uint8_t(0x21 + i);
    std::memcpy(next.bytes.data() + 160, old.bytes.data() + 144, 16);
    put<std::uint64_t>(next.bytes.data() + 176, 7);
    next.identity_crc();
}
TEST(BoltXchannel, LiteralPrefixUserOpaqueFieldsAndRefusalToConsume) {
    Fixture b;
    std::memcpy(b.bytes.data() + 208, kUser.data(), 24);
    const auto f = b.file();
    auto c = xc::begin(f);
    xc::Record r{};
    for (std::uint32_t i = 0; i < 10; ++i) {
        ASSERT_EQ(xc::peek(f, b.region(), c, r), Status::Ok);
        EXPECT_EQ(c.offset, 208u); EXPECT_EQ(c.record_index, 0u);
    }
    EXPECT_EQ(r.payload.data, b.bytes.data() + 224);
    EXPECT_EQ(r.payload.size, 3u); EXPECT_EQ(r.payload.data[2], 0xcc);
    EXPECT_EQ(r.message_type, 0x1234); EXPECT_EQ(r.user_meta_u64, 0x0102030405060708u);
    auto wrong = r; ++wrong.next_offset;
    EXPECT_EQ(xc::consume(f, b.region(), c, wrong), Status::Invalid);
    EXPECT_EQ(c.offset, 208u);
    ASSERT_EQ(xc::consume(f, b.region(), c, r), Status::Ok);
    EXPECT_EQ(c.offset, 232u); EXPECT_EQ(c.record_index, 1u);
    EXPECT_EQ(xc::peek(f, b.region(), c, r), Status::Idle);
    EXPECT_EQ(xc::consume(f, b.region(), c, wrong), Status::Idle);
}
TEST(BoltXchannel, EveryPrefixTruncationPreservesOutputAndRealV3Refusal) {
    Fixture b;
    xc::File out{}; out.sequence = 123;
    for (std::uint32_t n = 0; n < 208; ++n) {
        EXPECT_EQ(xc::admit_file({b.bytes.data(), n}, kExpected, kLimits, out), Status::Truncated);
        EXPECT_EQ(out.sequence, 123u);
    }
    b.bytes[56] = 3; b.bytes[4] = 128;
    EXPECT_EQ(xc::admit_file({b.bytes.data(), 144}, kExpected, kLimits, out), Status::Unsupported);
    EXPECT_EQ(out.sequence, 123u);
    b.bytes[56] = 4;
    EXPECT_EQ(xc::admit_file({b.bytes.data(), 144}, kExpected, kLimits, out), Status::Truncated);
}
TEST(BoltXchannel, PublicationAndIdentityReservedFields) {
    Fixture b;
    xc::File out{}; out.sequence = 777;
    for (const auto offset : {144u, 184u, 193u, 196u, 200u, 114u}) {
        b.bytes[offset] ^= 1;
        EXPECT_EQ(xc::admit_file({b.bytes.data(),208}, kExpected,kLimits,out), Status::Corrupt);
        EXPECT_EQ(out.sequence, 777u);
        b.bytes[offset] ^= 1;
    }
    b.bytes[192] = 0;
    put<std::uint64_t>(b.bytes.data() + 32, UINT64_MAX);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208}, kExpected,kLimits,out), Status::Prepared);
    EXPECT_EQ(out.sequence, 777u);
    b.bytes[192] = 2;
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208}, kExpected,kLimits,out), Status::Corrupt);
    b = Fixture{}; b.bytes[120] = 0xa5; // additive reserved2 must be ignored
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Ok);
    b.bytes[113] = 1;
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Ok);
    b.bytes[113] = 2;
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Corrupt);
}
TEST(BoltXchannel, AdmissionUnsupportedGeometryGenerationAndBase) {
    Fixture b; xc::File out{};
    for (const auto offset : {58u, 59u, 60u, 64u}) {
        const auto saved = b.bytes[offset]; b.bytes[offset] = 2;
        EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Unsupported);
        b.bytes[offset] = saved;
    }
    EXPECT_EQ(xc::admit_file({b.bytes.data()+1,207},kExpected,kLimits,out), Status::Invalid);
    EXPECT_EQ(xc::admit_file({nullptr,208},kExpected,kLimits,out), Status::Invalid);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},{1,68,0,false},kLimits,out), Status::Gap);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},{0,69,0,false},kLimits,out), Status::Recreated);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},{0,68,1,true},kLimits,out), Status::Gap);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,{4096,2048,10},out), Status::Capacity);
    put<std::uint32_t>(b.bytes.data()+48,4097);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Corrupt);
    put<std::uint32_t>(b.bytes.data()+48,128);
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out), Status::Corrupt);
}
TEST(BoltXchannel, UncommittedDirtyHeaderEveryRecordTruncationAndCommitFlag) {
    Fixture b; const auto f = b.file(); auto c = xc::begin(f); xc::Record out{};
    out.offset = 999;
    std::memset(b.bytes.data()+208,0xff,16); b.bytes[208] = 0;
    EXPECT_EQ(xc::peek(f,b.region(209),c,out),Status::Idle);
    EXPECT_EQ(out.offset,999u); EXPECT_EQ(c.offset,208u);
    std::memcpy(b.bytes.data()+208,kUser.data(),24);
    for (std::uint32_t n = 208; n < 232; ++n) {
        EXPECT_EQ(xc::peek(f,b.region(n),c,out),Status::Truncated);
        EXPECT_EQ(out.offset,999u); EXPECT_EQ(c.offset,208u);
    }
    b.bytes[208] = 2;
    EXPECT_EQ(xc::peek(f,b.region(),c,out),Status::Corrupt);
    b.bytes[208] = 1;
    for (const auto kind : {0u,4u,255u}) {
        b.bytes[209] = std::uint8_t(kind);
        EXPECT_EQ(xc::peek(f,b.region(),c,out),Status::Corrupt);
    }
    EXPECT_EQ(out.offset,999u);
}
TEST(BoltXchannel, ShorterReservationUsesActualSentinelAndBounds) {
    Fixture b; std::memcpy(b.bytes.data()+208,kUser.data(),24);
    // actual length3 => sentinel232; stale reservation sentinel at288 irrelevant.
    b.bytes[233] = 1; b.bytes[289] = 1;
    auto f = b.file(); auto c = xc::begin(f); xc::Record r{};
    ASSERT_EQ(xc::peek(f,b.region(),c,r),Status::Ok);
    ASSERT_EQ(xc::consume(f,b.region(),c,r),Status::Ok);
    EXPECT_EQ(c.offset,232u); EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Idle);
    c = xc::begin(f); f.payload_cap = 2;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Capacity); EXPECT_EQ(c.offset,208u);
    f.payload_cap = 2048; f.mtu = 2;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt);
    f.mtu = 0;
    put<std::uint32_t>(b.bytes.data()+212,UINT32_MAX);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt);
    put<std::uint32_t>(b.bytes.data()+212,3888-16);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt); // leaves no Roll room
    put<std::uint32_t>(b.bytes.data()+212,0);
    ASSERT_EQ(xc::peek(f,b.region(),c,r),Status::Ok); EXPECT_EQ(r.payload.size,0u);
    c.record_index = UINT64_MAX;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Overflow);
}
TEST(BoltXchannel, OrderedSkipExplicitRegionMappingAndBadSkip) {
    Fixture b; b.bytes[208]=1; b.bytes[209]=2;
    put<std::uint32_t>(b.bytes.data()+212,3872); // 4096-208-16
    std::memcpy(b.bytes.data()+4096,kUser.data(),24);
    auto f=b.file(); auto c=xc::begin(f); xc::Record r{};
    ASSERT_EQ(xc::peek(f,b.region(224),c,r),Status::Ok);
    EXPECT_EQ(r.kind,xc::Kind::Skip); EXPECT_EQ(r.payload.data,nullptr);
    ASSERT_EQ(xc::consume(f,b.region(224),c,r),Status::Ok);
    EXPECT_EQ(c.offset,4096u); EXPECT_EQ(c.record_index,0u);
    EXPECT_EQ(xc::required_region_offset(f,c),4096u);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::NeedRegion);
    ASSERT_EQ(xc::peek(f,b.region(4096,4096),c,r),Status::Ok);
    ASSERT_EQ(xc::consume(f,b.region(4096,4096),c,r),Status::Ok);
    EXPECT_EQ(c.record_index,1u);
    c=xc::begin(f); put<std::uint32_t>(b.bytes.data()+212,3864);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt);
}
TEST(BoltXchannel, LiteralRollAndExactSuccessorAdmission) {
    Fixture b,next; seed_roll(b,next);
    const auto f=b.file({7,68,900,true}), successor=next.file({8,68,1000,true});
    auto c=xc::begin(f); c.record_index=1000; xc::Record r{};
    ASSERT_EQ(xc::peek(f,b.region(),c,r),Status::Ok);
    EXPECT_EQ(r.kind,xc::Kind::Roll); EXPECT_EQ(r.roll.next_base_record_index,1000u);
    EXPECT_EQ(xc::consume(f,b.region(),c,r),Status::Invalid);
    xc::Cursor result{};
    ASSERT_EQ(xc::follow_successor(f,b.region(),c,r,successor,result),Status::Ok);
    EXPECT_EQ(result.offset,208u); EXPECT_EQ(result.record_index,1000u);
    EXPECT_EQ(c.offset,208u);
    for (const auto offset : {209u,212u,224u,232u,248u,256u,260u}) {
        b.bytes[offset]^=0x40;
        EXPECT_NE(xc::peek(f,b.region(),c,r),Status::Ok);
        EXPECT_EQ(c.offset,208u);
        b.bytes[offset]^=0x40;
    }
}
TEST(BoltXchannel, WrongSuccessorPreparedGenerationBaseAndFinalCount) {
    Fixture b,next; seed_roll(b,next);
    auto f=b.file({7,68,900,true}); auto successor=next.file({8,68,1000,true});
    auto c=xc::begin(f); c.record_index=1000; xc::Record r{};
    ASSERT_EQ(xc::peek(f,b.region(),c,r),Status::Ok);
    xc::Cursor out{}; out.offset=999;
    auto bad=successor; bad.instance.bytes[0]^=1;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.predecessor.bytes[0]^=1;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.predecessor_sequence=6;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.sequence=9;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.generation=99;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Recreated);
    bad=successor; bad.base_record_index=1001;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Gap);
    bad=successor; bad.region_size=8192;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.mtu=99;
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    bad=successor; bad.name[0]='!';
    EXPECT_EQ(xc::follow_successor(f,b.region(),c,r,bad,out),Status::Corrupt);
    EXPECT_EQ(out.offset,999u);
    next.bytes[192]=0;
    EXPECT_EQ(xc::admit_file({next.bytes.data(),208},{8,68,0,false},kLimits,bad),Status::Prepared);
    put<std::uint64_t>(b.bytes.data()+24,101);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Gap);
    put<std::uint64_t>(b.bytes.data()+24,UINT64_MAX);
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Overflow);
}
TEST(BoltXchannel, ExtensionSemanticCorruptionAndAllRollTruncations) {
    Fixture b; xc::File out{};
    std::memset(b.bytes.data()+144,0,16); b.identity_crc();
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out),Status::Corrupt);
    b=Fixture{}; b.bytes[176]=0; b.identity_crc(); // one genesis sentinel missing
    EXPECT_EQ(xc::admit_file({b.bytes.data(),208},kExpected,kLimits,out),Status::Corrupt);
    b=Fixture{}; b.bytes[0]=0; b.bytes[56]=255;
    EXPECT_EQ(xc::admit_file({b.bytes.data(),1},kExpected,kLimits,out),Status::Idle);
    b.bytes[0]=2;
    EXPECT_EQ(xc::admit_file({b.bytes.data(),1},kExpected,kLimits,out),Status::Corrupt);
    Fixture next; b=Fixture{}; seed_roll(b,next);
    auto f=b.file({7,68,900,true}); auto c=xc::begin(f); c.record_index=1000;
    xc::Record r{}; r.offset=999;
    for (std::uint32_t n=208;n<264;++n) {
        EXPECT_EQ(xc::peek(f,b.region(n),c,r),Status::Truncated);
        EXPECT_EQ(r.offset,999u); EXPECT_EQ(c.offset,208u);
    }
    std::memset(b.bytes.data()+232,0,16);
    put(b.bytes.data()+256,reference_crc(b.bytes.data()+224,32));
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt);
}
TEST(BoltXchannel, CursorMappingValidationAndAbsoluteOffsetOverflow) {
    Fixture b; const auto f=b.file(); auto c=xc::begin(f); xc::Record r{};
    c.instance.bytes[0]^=1;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Invalid);
    c=xc::begin(f); ++c.offset;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Invalid);
    c=xc::begin(f); c.offset=4088;
    EXPECT_EQ(xc::peek(f,b.region(),c,r),Status::Corrupt);
    c=xc::begin(f);
    EXPECT_EQ(xc::peek(f,{{b.bytes.data()+1,4095},0},c,r),Status::Invalid);
    EXPECT_EQ(xc::peek(f,{{b.bytes.data(),4096},1},c,r),Status::Invalid);
    EXPECT_EQ(xc::peek(f,{{b.bytes.data(),8192},0},c,r),Status::Invalid);
    b.bytes[208]=1; b.bytes[209]=2; put<std::uint32_t>(b.bytes.data()+212,3872);
    constexpr auto last_region=UINT64_MAX-4095;
    c.offset=last_region+208;
    EXPECT_EQ(xc::peek(f,{{b.bytes.data(),4096},last_region},c,r),Status::Overflow);
    EXPECT_EQ(c.offset,last_region+208);
}
TEST(BoltXchannel, DelayedReleaseCommitPublishesHeaderAndPayload) {
    Fixture b; auto f=b.file(); auto c=xc::begin(f); xc::Record r{};
    std::atomic<bool> go{false};
    std::thread writer([&] {
        bool proceed=false;
        for (std::uint32_t i=0;i<10000000;++i) {
            if (go.load(std::memory_order_relaxed)) { proceed=true; break; }
            if (i%1024==0) std::this_thread::yield();
        }
        if (!proceed) return;
        std::memcpy(b.bytes.data()+209,kUser.data()+1,23);
        std::atomic_ref<std::uint8_t>(b.bytes[208]).store(1,std::memory_order_release);
    });
    bool idle=true;
    for (std::uint32_t i=0;i<1000;++i) idle &= xc::peek(f,b.region(),c,r)==Status::Idle;
    go.store(true,std::memory_order_relaxed);
    Status observed=Status::Idle; std::uint8_t payload=0; std::uint64_t meta=0;
    for (std::uint32_t i=0;i<10000000;++i) {
        observed=xc::peek(f,b.region(),c,r);
        if (observed!=Status::Idle) {
            if (observed==Status::Ok) { payload=r.payload.data[2]; meta=r.user_meta_u64; }
            break;
        }
        if (i%1024==0) std::this_thread::yield();
    }
    writer.join(); // assertions happen after join; payload snapshot was before.
    EXPECT_TRUE(idle); EXPECT_EQ(observed,Status::Ok);
    EXPECT_EQ(payload,0xcc); EXPECT_EQ(meta,0x0102030405060708u);
    EXPECT_EQ(c.offset,208u);
}
} // namespace
