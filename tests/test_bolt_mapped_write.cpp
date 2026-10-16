#include "bolt/io/bolt_file_io.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
namespace {
using namespace bolt::io;
std::string unique_path() {
    static std::atomic<unsigned> serial{0};
    return (std::filesystem::temp_directory_path()/("bolt-mapped-write-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+
        std::to_string(serial.fetch_add(1)))).string();
}
TEST(MappedWrite, ExactExtentFlushAndExistingPathProtection) {
    const auto path=unique_path();const auto unit=std::max(io_page_size(),io_map_granularity());
    const auto bytes=std::uint64_t{4}*unit;WritableMappedFile m{};
    ASSERT_EQ(mapped_write_create(path.c_str(),bytes,false,true,&m),FileIoStatus::kOk);
    ASSERT_NE(m.base,nullptr);EXPECT_EQ(m.size,bytes);EXPECT_EQ(m.page,io_page_size());
    EXPECT_EQ(m.granularity,io_map_granularity());EXPECT_EQ(std::filesystem::file_size(path),bytes);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(m.base)%io_page_size(),0u);
    const std::uint8_t literal[]={0,0xff,0x41,0,0x5a};
    std::memcpy(m.base+unit-2,literal,sizeof(literal));
    EXPECT_EQ(mapped_write_flush(&m,unit-2,sizeof(literal)),FileIoStatus::kOk);
    EXPECT_EQ(mapped_write_flush(&m,bytes,0),FileIoStatus::kOk);
    EXPECT_NE(mapped_write_flush(&m,bytes,1),FileIoStatus::kOk);
    EXPECT_NE(mapped_write_flush(&m,UINT64_MAX,2),FileIoStatus::kOk);
    mapped_write_close(&m);mapped_write_close(&m);
    WritableMappedFile refused{};
    EXPECT_NE(mapped_write_create(path.c_str(),bytes*2,false,false,&refused),FileIoStatus::kOk);
    EXPECT_EQ(std::filesystem::file_size(path),bytes);
    auto* file=std::fopen(path.c_str(),"rb");ASSERT_NE(file,nullptr);
    ASSERT_EQ(std::fseek(file,static_cast<long>(unit-2),SEEK_SET),0);std::uint8_t actual[5]{};
    EXPECT_EQ(std::fread(actual,1,sizeof(actual),file),sizeof(actual));
    EXPECT_EQ(std::memcmp(actual,literal,sizeof(literal)),0);EXPECT_EQ(std::fclose(file),0);
    EXPECT_EQ(std::remove(path.c_str()),0);
}
TEST(MappedWrite, InvalidGeometryLeavesNoFileAndEphemeralCloseDeletes) {
    const auto path=unique_path();const auto unit=std::max(io_page_size(),io_map_granularity());
    WritableMappedFile m{};
    EXPECT_NE(mapped_write_create(path.c_str(),0,false,false,&m),FileIoStatus::kOk);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_NE(mapped_write_create(path.c_str(),unit+1,false,false,&m),FileIoStatus::kOk);
    EXPECT_FALSE(std::filesystem::exists(path));
    ASSERT_EQ(mapped_write_create(path.c_str(),unit,true,true,&m),FileIoStatus::kOk);
    m.base[unit-1]=0x7f;mapped_write_close(&m);EXPECT_FALSE(std::filesystem::exists(path));
}
} // namespace
