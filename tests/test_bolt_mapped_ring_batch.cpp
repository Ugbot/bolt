#include "bolt/io/bolt_mapped_ring_batch.h"
#include "bolt/io/bolt_file_io.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <string>

namespace {
using namespace bolt;
using namespace bolt::io;
using S=RingStatus;
struct Owner {
    MappedRing* ring=nullptr;
    std::string name;
    Owner() {
        static std::atomic<unsigned> serial{0};
        name=(std::filesystem::temp_directory_path()/("bolt-ring-batch-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+
            std::to_string(serial.fetch_add(1)))).string();
    }
    ~Owner() { if(ring) EXPECT_EQ(mapped_ring_close(&ring),S::kOk); }
    void open() {
        const auto unit=std::max(io_page_size(),io_map_granularity());
        RingOptions options{4ull*unit,io_page_size(),1,true,true,false,nullptr,nullptr,nullptr};
        ASSERT_EQ(mapped_ring_open(name.c_str(),options,&ring),S::kOk);
    }
};
struct Input {
    std::uint64_t numbers[3]{0,UINT64_MAX,0x0123456789abcdefull};
    std::int32_t offsets[4]{0,3,3,8};
    std::uint8_t bytes[8]{0,0xff,0x41,0x80,0,1,2,0};
    std::uint8_t validity=5;
    BoltColumn columns[2]{},offset_column{};
    BoltField fields[2]{};
    BoltBatch batch{};
    Input() {
        BoltBatch::init_empty(&batch);batch.columns[0]=batch.columns[1]=columns;
        batch.num_cols=2;batch.num_rows=3;batch.schema.set_storage(fields,2);
        (void)batch.schema.add_field("number",BoltType::UInt64,false);
        (void)batch.schema.add_field("binary",BoltType::Binary,true);
        columns[0]=BoltColumn::make_flat(numbers,nullptr,3,BoltType::UInt64);
        offset_column=BoltColumn::make_flat(offsets,nullptr,4,BoltType::Int32);
        columns[1]=BoltColumn::make_empty();columns[1].data=bytes;columns[1].length=3;
        columns[1].type=BoltType::Binary;columns[1].format=ColumnFormat::VarBinary;
        columns[1].dict_child=&offset_column;columns[1].validity=&validity;
    }
};
wire::FrameMeta meta() {
    wire::FrameMeta m{};m.kind=wire::FrameOpKind::kBatch;m.flags=0x1357;m.first_pos=99;m.crc_seed=17;
    return m;
}
TEST(MappedRingBatch, NullableLiteralExtentsAndExplicitRetryLease) {
    Owner owner;owner.open();Input input;Arena arena;
    ASSERT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kOk);
    BoltBatch batch{};RingView lease{};wire::FrameView frame{};
    ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,&frame,17),S::kOk);
    EXPECT_EQ(frame.header.flags,0x1357);EXPECT_EQ(frame.header.first_pos,99u);
    ASSERT_EQ(batch.num_rows,3);ASSERT_EQ(batch.num_cols,2u);
    EXPECT_EQ(std::memcmp(batch.col(0).data,input.numbers,sizeof(input.numbers)),0);
    EXPECT_TRUE(batch.col(1).is_null(1));EXPECT_FALSE(batch.col(1).is_null(0));
    EXPECT_EQ(std::memcmp(batch.col(1).data,input.bytes,sizeof(input.bytes)),0);
    EXPECT_EQ(std::memcmp(batch.col(1).dict_child->data,input.offsets,sizeof(input.offsets)),0);
    const auto sequence=lease.sequence;
    EXPECT_EQ(mapped_ring_close(&owner.ring),S::kBusy);
    ASSERT_EQ(mapped_ring_unpin(owner.ring,lease),S::kOk);
    arena.reset();
    ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,nullptr,17),S::kOk);
    EXPECT_EQ(lease.sequence,sequence);
    ASSERT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
    EXPECT_EQ(mapped_ring_peek(owner.ring,0,&lease),S::kEmpty);
}
TEST(MappedRingBatch, ReusedOutputShellAdvancesGenerationAndRetryKeepsIdentity) {
    Owner owner;owner.open();Input input;Arena arena;
    BoltBatch output{};RingView lease{};
    for(std::uint64_t sequence=1;sequence<=2;++sequence) {
        input.numbers[0]=sequence*111;
        ASSERT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kOk);
        arena.reset();
        ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&output,&lease,nullptr,17),S::kOk);
        EXPECT_EQ(output.generation[0],sequence);EXPECT_EQ(output.generation[1],sequence);
        EXPECT_EQ(static_cast<const std::uint64_t*>(output.col(0).data)[0],sequence*111);
        ASSERT_EQ(mapped_ring_unpin(owner.ring,lease),S::kOk);
        const auto prior_rows=output.num_rows;
        EXPECT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&output,&lease,nullptr,18),S::kCorrupt);
        EXPECT_EQ(output.num_rows,prior_rows);
        EXPECT_EQ(output.generation[0],sequence);EXPECT_EQ(output.generation[1],sequence);
        arena.reset();
        ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&output,&lease,nullptr,17),S::kOk);
        EXPECT_EQ(output.generation[0],sequence);EXPECT_EQ(output.generation[1],sequence);
        EXPECT_EQ(lease.sequence,sequence);
        ASSERT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
    }
}
TEST(MappedRingBatch, WrongFrameSeedUnpinsWithoutConsumingOrChangingOutputs) {
    Owner owner;owner.open();Input input;Arena arena;
    ASSERT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kOk);
    BoltBatch batch{};batch.num_rows=123;RingView lease{};lease.sequence=456;
    EXPECT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,nullptr,18),S::kCorrupt);
    EXPECT_EQ(batch.num_rows,123);EXPECT_EQ(lease.sequence,456u);
    ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,nullptr,17),S::kOk);
    EXPECT_EQ(lease.sequence,1u);EXPECT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
}
TEST(MappedRingBatch, ExhaustedDescriptorArenaRefusesWithoutConsumingThenRetries) {
    Owner owner;owner.open();Input input;
    ASSERT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kOk);
    ArenaConfig config{};config.initial_block_size=64;config.max_block_size=64;
    Arena tiny{config};ASSERT_NE(tiny.allocate(64,64),nullptr);tiny.set_growth_enabled(false);
    const auto used=tiny.total_allocated();
    BoltBatch batch{};batch.num_rows=123;RingView lease{};lease.sequence=456;
    wire::FrameView frame{};frame.frame_len=789;
    EXPECT_EQ(mapped_ring_peek_batch(owner.ring,0,&tiny,&batch,&lease,&frame,17),S::kDecodeRefused);
    EXPECT_EQ(tiny.total_allocated(),used);EXPECT_EQ(tiny.allocate(1,1),nullptr);
    EXPECT_EQ(batch.num_rows,123);EXPECT_EQ(lease.sequence,456u);EXPECT_EQ(frame.frame_len,789u);
    Arena adequate;
    ASSERT_NE(adequate.allocate(65536,64),nullptr);adequate.reset();adequate.set_growth_enabled(false);
    ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&adequate,&batch,&lease,&frame,17),S::kOk);
    EXPECT_EQ(lease.sequence,1u);EXPECT_EQ(batch.num_rows,3);
    EXPECT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
}
TEST(MappedRingBatch, FullRetainsCallerBatchAndFailedEncodeCancelsReservation) {
    Owner owner;owner.open();Input input;
    auto invalid=meta();invalid.kind=wire::FrameOpKind::kNone;
    EXPECT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,invalid),S::kUnsupported);
    EXPECT_EQ(mapped_ring_snapshot(owner.ring).next_sequence,1u);
    std::uint32_t accepted=0;
    for(;accepted<16384;++accepted) {
        const auto status=mapped_ring_write_batch(owner.ring,&input.batch,meta());
        if(status==S::kFull)break;
        ASSERT_EQ(status,S::kOk);
    }
    ASSERT_GT(accepted,0u);ASSERT_LT(accepted,16384u);
    const auto before=mapped_ring_snapshot(owner.ring);
    EXPECT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kFull);
    EXPECT_EQ(mapped_ring_snapshot(owner.ring).committed_position,before.committed_position);
    EXPECT_EQ(input.numbers[1],UINT64_MAX);EXPECT_EQ(input.bytes[1],0xff);
    Arena arena;BoltBatch batch{};RingView lease{};
    for(std::uint32_t i=0;i<accepted;++i) {
        arena.reset();ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,nullptr,17),S::kOk);
        EXPECT_EQ(lease.sequence,std::uint64_t{i}+1);ASSERT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
    }
    ASSERT_EQ(mapped_ring_write_batch(owner.ring,&input.batch,meta()),S::kOk);
    arena.reset();ASSERT_EQ(mapped_ring_peek_batch(owner.ring,0,&arena,&batch,&lease,nullptr,17),S::kOk);
    EXPECT_EQ(lease.sequence,std::uint64_t{accepted}+1);EXPECT_EQ(mapped_ring_release(owner.ring,lease),S::kOk);
}
} // namespace
