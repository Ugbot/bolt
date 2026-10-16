#include "bolt/io/bolt_mapped_ring.h"
#include "bolt/io/bolt_file_io.h"
#include "bolt/bolt_budget.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace bolt::io;
using S=RingStatus;
std::string path() {
    static std::atomic<unsigned> serial{0};
    return (std::filesystem::temp_directory_path()/ ("bolt-ring-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+
        std::to_string(serial.fetch_add(1)))).string();
}
struct Owner {
    std::string name=path();MappedRing* ring=nullptr;
    ~Owner(){if(ring)EXPECT_EQ(mapped_ring_close(&ring),S::kOk);std::remove(name.c_str());}
    RingOptions options(std::uint32_t readers=1) const {
        const auto unit=std::max(io_page_size(),io_map_granularity());
        return {std::uint64_t{8}*unit,std::uint64_t{2}*io_page_size(),readers,false,true,false,nullptr,nullptr,nullptr};
    }
    void open(std::uint32_t readers=1) {ASSERT_EQ(mapped_ring_open(name.c_str(),options(readers),&ring),S::kOk);}
};
void put(MappedRing* ring,const void* data,std::uint64_t n) {
    RingReservation r{};ASSERT_EQ(mapped_ring_reserve(ring,n,&r),S::kOk);
    if(n)std::memcpy(r.data,data,static_cast<std::size_t>(n));
    ASSERT_EQ(mapped_ring_commit(ring,r),S::kOk);
}
void consume(MappedRing* ring,std::uint32_t reader,const void* data,std::uint64_t n,std::uint64_t sequence) {
    RingView v{};ASSERT_EQ(mapped_ring_peek(ring,reader,&v),S::kOk);
    EXPECT_EQ(v.bytes,n);EXPECT_EQ(v.sequence,sequence);
    if(v.bytes==n && n)EXPECT_EQ(std::memcmp(v.data,data,static_cast<std::size_t>(n)),0);
    EXPECT_EQ(mapped_ring_release(ring,v),S::kOk);
}
TEST(MappedRing, PackedLiteralBinaryZeroAndMultipageRecords) {
    Owner o;o.open();ASSERT_NE(o.ring,nullptr);const auto s=mapped_ring_snapshot(o.ring);
    EXPECT_EQ(s.page_bytes,io_page_size());EXPECT_EQ(s.mapping_granularity,io_map_granularity());
    EXPECT_EQ(s.capacity_bytes,o.options().capacity_bytes);EXPECT_EQ(std::filesystem::file_size(o.name),s.capacity_bytes);
    const std::uint8_t bytes[]={0,0xff,0x41,0,0x7f};
    put(o.ring,nullptr,0);put(o.ring,bytes,sizeof(bytes));
    auto after=mapped_ring_snapshot(o.ring);EXPECT_EQ(after.committed_position%64,0u);
    EXPECT_LT(after.committed_position,std::uint64_t{s.page_bytes});
    consume(o.ring,0,nullptr,0,1);consume(o.ring,0,bytes,sizeof(bytes),2);
    std::vector<std::uint8_t> large(io_page_size()+117);
    for(std::size_t i=0;i<large.size();++i)large[i]=static_cast<std::uint8_t>(i*17);
    put(o.ring,large.data(),large.size());consume(o.ring,0,large.data(),large.size(),3);
    RingView empty{};EXPECT_EQ(mapped_ring_peek(o.ring,0,&empty),S::kEmpty);
}
TEST(MappedRing, PublicationCancelStaleTokensAndCloseBusy) {
    Owner o;o.open();RingReservation r{};ASSERT_EQ(mapped_ring_reserve(o.ring,8,&r),S::kOk);
    std::memset(r.data,0x5a,8);RingView v{};EXPECT_EQ(mapped_ring_peek(o.ring,0,&v),S::kEmpty);
    EXPECT_EQ(mapped_ring_close(&o.ring),S::kBusy);ASSERT_NE(o.ring,nullptr);
    EXPECT_EQ(mapped_ring_cancel(o.ring,r),S::kOk);EXPECT_NE(mapped_ring_commit(o.ring,r),S::kOk);
    EXPECT_EQ(mapped_ring_peek(o.ring,0,&v),S::kEmpty);
    RingReservation next{};ASSERT_EQ(mapped_ring_reserve(o.ring,8,&next),S::kOk);
    EXPECT_EQ(next.sequence,1u);EXPECT_NE(next.token,r.token);std::memset(next.data,0x33,8);
    ASSERT_EQ(mapped_ring_commit(o.ring,next),S::kOk);EXPECT_NE(mapped_ring_commit(o.ring,next),S::kOk);
    ASSERT_EQ(mapped_ring_peek(o.ring,0,&v),S::kOk);EXPECT_EQ(mapped_ring_close(&o.ring),S::kBusy);
    EXPECT_EQ(mapped_ring_release(o.ring,v),S::kOk);EXPECT_NE(mapped_ring_release(o.ring,v),S::kOk);
}
TEST(MappedRing, FullRefusalIsLosslessAndBothReadersFenceReuse) {
    Owner o;o.open(2);std::uint64_t accepted=0;RingReservation r{};
    const auto cap=mapped_ring_snapshot(o.ring).capacity_bytes;
    for(std::uint64_t i=0;i<cap/64+1;++i) {
        const auto status=mapped_ring_reserve(o.ring,8,&r);if(status==S::kFull)break;
        ASSERT_EQ(status,S::kOk);++accepted;std::memcpy(r.data,&accepted,8);ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);
    }
    ASSERT_GT(accepted,0u);const auto before=mapped_ring_snapshot(o.ring);
    EXPECT_EQ(mapped_ring_reserve(o.ring,8,&r),S::kFull);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).next_sequence,before.next_sequence);
    for(std::uint64_t i=1;i<=accepted;++i)consume(o.ring,0,&i,8,i);
    EXPECT_EQ(mapped_ring_reserve(o.ring,8,&r),S::kFull);
    for(std::uint64_t i=1;i<=accepted;++i)consume(o.ring,1,&i,8,i);
    const std::uint64_t value=accepted+1;put(o.ring,&value,8);
    consume(o.ring,0,&value,8,value);consume(o.ring,1,&value,8,value);
}
TEST(MappedRing, WrapPaddingAndGenerationsPreserveCopiedPayloads) {
    Owner o;o.open();std::vector<std::uint8_t> data(io_page_size()+19);
    for(std::uint64_t sequence=1;sequence<=200;++sequence) {
        const auto size=sequence%3==0?data.size():static_cast<std::size_t>(sequence%173+1);
        for(std::size_t i=0;i<size;++i)data[i]=static_cast<std::uint8_t>(sequence+i*29);
        put(o.ring,data.data(),size);consume(o.ring,0,data.data(),size,sequence);
    }
    EXPECT_GT(mapped_ring_snapshot(o.ring).committed_position,o.options().capacity_bytes);
}
TEST(MappedRing, OverwriteReportsExactGapAndNeverOverwritesLiveView) {
    Owner o;o.open();RingReservation r{};std::uint64_t accepted=0;
    for(std::uint64_t i=0;i<o.options().capacity_bytes/64+1;++i) {
        auto st=mapped_ring_reserve(o.ring,8,&r);if(st==S::kFull)break;
        ASSERT_EQ(st,S::kOk);++accepted;std::memcpy(r.data,&accepted,8);ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);
    }
    RingView pinned{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&pinned),S::kOk);
    EXPECT_EQ(mapped_ring_reserve_overwrite(o.ring,8,&r),S::kFull);
    std::uint64_t original=0;std::memcpy(&original,pinned.data,8);EXPECT_EQ(original,1u);
    ASSERT_EQ(mapped_ring_release(o.ring,pinned),S::kOk);
    // Fill the space released by event 1 without advancing this reader again.
    for(std::uint64_t i=0;i<o.options().capacity_bytes/64+1;++i) {
        const auto status=mapped_ring_reserve(o.ring,8,&r);if(status==S::kFull)break;
        ASSERT_EQ(status,S::kOk);++accepted;std::memcpy(r.data,&accepted,8);
        ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);
    }
    ASSERT_EQ(mapped_ring_reserve(o.ring,8,&r),S::kFull);
    ASSERT_EQ(mapped_ring_reserve_overwrite(o.ring,8,&r),S::kOk);
    const auto last=accepted+1;std::memcpy(r.data,&last,8);ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);
    RingView gap{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&gap),S::kGap);
    EXPECT_EQ(gap.lost_begin,2u);EXPECT_EQ(gap.lost_end,last);
    RingView next{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&next),S::kOk);
    EXPECT_EQ(next.sequence,last);std::uint64_t value=0;std::memcpy(&value,next.data,8);
    EXPECT_EQ(value,last);EXPECT_EQ(mapped_ring_release(o.ring,next),S::kOk);
}
TEST(MappedRing, LogicalLimitsAndGeometryRefuseWithoutMutation) {
    Owner o;auto opt=o.options();opt.capacity_bytes+=1;
    EXPECT_EQ(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kBadArg);EXPECT_EQ(o.ring,nullptr);
    o.open();const auto initial=mapped_ring_snapshot(o.ring);
    EXPECT_NE(mapped_ring_set_limit(o.ring,initial.capacity_bytes+initial.page_bytes),S::kOk);
    EXPECT_NE(mapped_ring_set_limit(o.ring,initial.capacity_bytes-1),S::kOk);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).active_limit_bytes,initial.active_limit_bytes);
    const auto reduced=initial.capacity_bytes-std::max(initial.page_bytes,initial.mapping_granularity);
    ASSERT_EQ(mapped_ring_set_limit(o.ring,reduced),S::kOk);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).active_limit_bytes,reduced);
    EXPECT_EQ(mapped_ring_set_limit(o.ring,initial.capacity_bytes),S::kOk);
    std::vector<std::uint8_t> bytes(io_page_size(),0xa5);RingReservation r{};
    for(std::uint64_t i=0;i<initial.capacity_bytes/64+1;++i) {
        const auto status=mapped_ring_reserve(o.ring,bytes.size(),&r);if(status==S::kFull)break;
        ASSERT_EQ(status,S::kOk);std::memcpy(r.data,bytes.data(),bytes.size());ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);
    }
    EXPECT_NE(mapped_ring_set_limit(o.ring,initial.page_bytes),S::kOk);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).active_limit_bytes,initial.capacity_bytes);
}
TEST(MappedRing, BudgetHierarchyRollbackStableAdmissionAndRelease) {
    const auto cap=std::uint64_t{8}*std::max(io_page_size(),io_map_granularity());
    bolt::Budget disk_parent("ring_disk_parent",cap-1),disk("ring_disk",cap*2,&disk_parent);
    bolt::Budget memory("ring_memory",64*1024*1024),files("ring_files",1);
    Owner o;auto opt=o.options();
    opt.disk_budget=&disk;opt.memory_budget=&memory;opt.file_budget=&files;
    EXPECT_NE(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kOk);EXPECT_EQ(o.ring,nullptr);
    EXPECT_EQ(disk.used(),0u);EXPECT_EQ(disk_parent.used(),0u);EXPECT_EQ(memory.used(),0u);EXPECT_EQ(files.used(),0u);
    disk_parent.set_limit(cap*2);files.set_limit(0);
    EXPECT_NE(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kOk);
    EXPECT_EQ(disk.used(),0u);EXPECT_EQ(memory.used(),0u);EXPECT_EQ(files.used(),0u);
    files.set_limit(1);memory.set_limit(0);
    EXPECT_NE(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kOk);
    EXPECT_EQ(disk.used(),0u);EXPECT_EQ(disk_parent.used(),0u);EXPECT_EQ(files.used(),0u);
    memory.set_limit(64*1024*1024);ASSERT_EQ(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kOk);
    const auto mem=memory.used();EXPECT_GT(mem,0u);EXPECT_EQ(disk.used(),cap);EXPECT_EQ(files.used(),1u);
    const std::uint64_t value=123;put(o.ring,&value,8);consume(o.ring,0,&value,8,1);
    EXPECT_EQ(mapped_ring_set_limit(o.ring,cap-io_page_size()),S::kOk);
    EXPECT_EQ(disk.used(),cap);EXPECT_EQ(memory.used(),mem);EXPECT_EQ(files.used(),1u);
    EXPECT_EQ(mapped_ring_close(&o.ring),S::kOk);EXPECT_EQ(o.ring,nullptr);
    EXPECT_EQ(disk.used(),0u);EXPECT_EQ(disk_parent.used(),0u);EXPECT_EQ(memory.used(),0u);EXPECT_EQ(files.used(),0u);
}
TEST(MappedRing, ExistingFileIsNeverTruncated) {
    Owner o;const std::uint8_t literal[]={0,0xff,0x6e,0x7f};
    auto* file=std::fopen(o.name.c_str(),"wb");ASSERT_NE(file,nullptr);
    ASSERT_EQ(std::fwrite(literal,1,sizeof(literal),file),sizeof(literal));ASSERT_EQ(std::fclose(file),0);
    EXPECT_NE(mapped_ring_open(o.name.c_str(),o.options(),&o.ring),S::kOk);EXPECT_EQ(o.ring,nullptr);
    EXPECT_EQ(std::filesystem::file_size(o.name),sizeof(literal));
    file=std::fopen(o.name.c_str(),"rb");ASSERT_NE(file,nullptr);std::uint8_t actual[4]{};
    EXPECT_EQ(std::fread(actual,1,sizeof(actual),file),sizeof(actual));EXPECT_EQ(std::memcmp(actual,literal,4),0);
    EXPECT_EQ(std::fclose(file),0);
}
TEST(MappedRing, FlushRetentionFencesWrapUntilExplicitFlush) {
    Owner o;auto opt=o.options();opt.retain_until_flush=true;
    ASSERT_EQ(mapped_ring_open(o.name.c_str(),opt,&o.ring),S::kOk);
    const auto cap=opt.capacity_bytes;RingReservation r{};std::uint64_t count=0;
    for(std::uint64_t i=0;i<cap/64+1;++i) {
        const auto status=mapped_ring_reserve(o.ring,8,&r);if(status==S::kFull)break;
        ASSERT_EQ(status,S::kOk);++count;std::memcpy(r.data,&count,8);
        ASSERT_EQ(mapped_ring_commit(o.ring,r),S::kOk);consume(o.ring,0,&count,8,count);
    }
    EXPECT_GT(count,0u);EXPECT_EQ(mapped_ring_reserve(o.ring,8,&r),S::kFull);
    const auto before=mapped_ring_snapshot(o.ring);EXPECT_EQ(before.durable_position,0u);
    ASSERT_EQ(mapped_ring_flush(o.ring),S::kOk);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).durable_position,before.committed_position);
    const std::uint64_t next=count+1;put(o.ring,&next,8);consume(o.ring,0,&next,8,next);
}
TEST(MappedRing, EphemeralReuseCannotCertifyLostUnsyncedPrefix) {
    Owner o;o.open();const auto cap=o.options().capacity_bytes;
    for(std::uint64_t i=1;i<=cap/64+100;++i){put(o.ring,&i,8);consume(o.ring,0,&i,8,i);}
    const auto before=mapped_ring_snapshot(o.ring);EXPECT_GT(before.committed_position,cap);
    EXPECT_EQ(mapped_ring_flush(o.ring),S::kUnsupported);
    EXPECT_EQ(mapped_ring_snapshot(o.ring).durable_position,before.durable_position);
}
TEST(MappedRing, ConcurrentProducerConsumerPublishesWholePayloads) {
    Owner o;o.open();constexpr std::uint64_t count=20000,attempts=20000000;
    std::atomic<bool> failed{false};std::thread consumer([&]{
        std::uint64_t expected=1;
        for(std::uint64_t attempt=0;attempt<attempts && expected<=count && !failed.load();++attempt) {
            RingView v{};const auto status=mapped_ring_peek(o.ring,0,&v);
            if(status==S::kEmpty){std::this_thread::yield();continue;}
            if(status!=S::kOk){failed.store(true);break;}
            std::uint64_t words[8]{};
            if(v.bytes!=sizeof(words) || v.sequence!=expected)failed.store(true);
            else {std::memcpy(words,v.data,sizeof(words));for(auto word:words)if(word!=expected)failed.store(true);}
            if(mapped_ring_release(o.ring,v)!=S::kOk)failed.store(true);
            ++expected;
        }
        if(expected!=count+1)failed.store(true);
    });
    std::uint64_t next=1;
    for(std::uint64_t attempt=0;attempt<attempts && next<=count && !failed.load();++attempt) {
        RingReservation r{};const auto status=mapped_ring_reserve(o.ring,64,&r);
        if(status==S::kFull){std::this_thread::yield();continue;}
        if(status!=S::kOk){failed.store(true);break;}
        std::uint64_t words[8];for(auto& word:words)word=next;std::memcpy(r.data,words,sizeof(words));
        if(mapped_ring_commit(o.ring,r)!=S::kOk)failed.store(true);
        ++next;
    }
    if(next!=count+1)failed.store(true);
    consumer.join();EXPECT_FALSE(failed.load());
    EXPECT_EQ(next,count+1);
}
void corrupt_file_bytes(const std::string& name,long offset,const void* bytes,std::size_t size) {
    auto* file=std::fopen(name.c_str(),"r+b");ASSERT_NE(file,nullptr);
    const auto sought=std::fseek(file,offset,SEEK_SET);
    EXPECT_EQ(sought,0);
    if(sought==0)EXPECT_EQ(std::fwrite(bytes,1,size,file),size);
    EXPECT_EQ(std::fclose(file),0);
}
TEST(MappedRing, PayloadCorruptionRefusesWithoutLeakingPin) {
    Owner o;o.open();const std::uint64_t value=0x0123456789abcdefULL;put(o.ring,&value,8);
    // The public packed-record geometry is a 64-byte header followed by bytes.
    const std::uint8_t changed=0x5a;corrupt_file_bytes(o.name,64,&changed,1);
    RingView view{};EXPECT_EQ(mapped_ring_peek(o.ring,0,&view),S::kCorrupt);
    EXPECT_EQ(mapped_ring_close(&o.ring),S::kOk);EXPECT_EQ(o.ring,nullptr);
}
TEST(MappedRing, HeaderCorruptionRefusesWithoutLeakingPin) {
    Owner o;o.open();const std::uint64_t value=73;put(o.ring,&value,8);
    const std::uint8_t destroyed_header[64]{};
    corrupt_file_bytes(o.name,0,destroyed_header,sizeof(destroyed_header));
    RingView view{};EXPECT_EQ(mapped_ring_peek(o.ring,0,&view),S::kCorrupt);
    EXPECT_EQ(mapped_ring_close(&o.ring),S::kOk);EXPECT_EQ(o.ring,nullptr);
}
TEST(MappedRing, RepeatedOverwritePreservesEarliestUnobservedGap) {
    Owner o;o.open();std::uint64_t accepted=0;RingReservation reservation{};
    const auto bound=o.options().capacity_bytes/64+1;
    for(unsigned round=0;round<3;++round) {
        for(std::uint64_t i=0;i<bound;++i) {
            const auto status=mapped_ring_reserve(o.ring,8,&reservation);
            if(status==S::kFull)break;
            ASSERT_EQ(status,S::kOk);++accepted;std::memcpy(reservation.data,&accepted,8);
            ASSERT_EQ(mapped_ring_commit(o.ring,reservation),S::kOk);
        }
        ASSERT_EQ(mapped_ring_reserve(o.ring,8,&reservation),S::kFull);
        ASSERT_EQ(mapped_ring_reserve_overwrite(o.ring,8,&reservation),S::kOk);
        ++accepted;std::memcpy(reservation.data,&accepted,8);
        ASSERT_EQ(mapped_ring_commit(o.ring,reservation),S::kOk);
    }
    RingView gap{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&gap),S::kGap);
    EXPECT_EQ(gap.lost_begin,1u);EXPECT_EQ(gap.lost_end,accepted);
    consume(o.ring,0,&accepted,8,accepted);
}
TEST(MappedRing, StaleLeaseCannotReleaseSubsequentLiveView) {
    Owner o;o.open();const std::uint64_t first=11,second=22;
    put(o.ring,&first,8);put(o.ring,&second,8);
    RingView old{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&old),S::kOk);
    ASSERT_EQ(mapped_ring_release(o.ring,old),S::kOk);
    RingView current{};ASSERT_EQ(mapped_ring_peek(o.ring,0,&current),S::kOk);
    EXPECT_NE(current.lease,old.lease);EXPECT_NE(mapped_ring_release(o.ring,old),S::kOk);
    EXPECT_EQ(mapped_ring_close(&o.ring),S::kBusy);ASSERT_NE(o.ring,nullptr);
    std::uint64_t actual=0;std::memcpy(&actual,current.data,8);EXPECT_EQ(actual,second);
    EXPECT_EQ(mapped_ring_release(o.ring,current),S::kOk);
    RingView empty{};EXPECT_EQ(mapped_ring_peek(o.ring,0,&empty),S::kEmpty);
}
} // namespace
