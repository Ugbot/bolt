#include "bolt/interop/arrow_owner.h"
#include "bolt/bolt_arena.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <cstring>
#include <atomic>
#include <thread>

namespace {
using namespace bolt::arrow;
ArrowOwner* fixture(std::uint32_t leases = 3, std::int64_t rows = 2) {
    ArrowOwner* o = nullptr;
    if (owner_create({1024 * 1024, 2, leases}, &o) != OwnerStatus::Ok) return nullptr;
    auto* arena = owner_build_arena(o);
    auto* values = arena->allocate_array<std::uint64_t>(2);
    auto* offsets = arena->allocate_array<std::int32_t>(3);
    auto* bytes = arena->allocate_array<unsigned char>(3);
    if (values == nullptr || offsets == nullptr || bytes == nullptr) { owner_release(o); return nullptr; }
    values[0] = UINT64_MAX; values[1] = UINT64_C(9223372036854775808);
    offsets[0] = 0; offsets[1] = 2; offsets[2] = 3;
    bytes[0] = 0; bytes[1] = 255; bytes[2] = 7;
    const void* root_buffers[]{nullptr};
    const void* value_buffers[]{nullptr, values};
    const void* byte_buffers[]{nullptr, offsets, bytes};
    ArrowSchema sv{}, sb{}, root{};
    sv.format = "L"; sv.name = "id"; sv.flags = 2;
    sb.format = "z"; sb.name = "raw"; sb.flags = 2;
    ArrowSchema* schemas[]{&sv, &sb};
    root.format = "+s"; root.name = ""; root.n_children = 2; root.children = schemas;
    ArrowArray av{}, ab{}, array{};
    av.length = rows; av.n_buffers = 2; av.buffers = value_buffers;
    ab.length = rows; ab.n_buffers = 3; ab.buffers = byte_buffers;
    ArrowArray* arrays[]{&av, &ab};
    array.length = rows; array.n_buffers = 1; array.buffers = root_buffers;
    array.n_children = 2; array.children = arrays;
    if (owner_publish(o, root, array) != OwnerStatus::Ok) { owner_release(o); return nullptr; }
    return o;
}
}

TEST(BoltArrowOwner, ExactSnapshotAndIndependentSchemaArrayRelease) {
    auto* o = fixture(); ASSERT_NE(o, nullptr);
    ArrowSchema s{}; ArrowArray a{};
    ASSERT_EQ(owner_export_array(o, &s, &a), OwnerStatus::Ok);
    EXPECT_STREQ(s.children[0]->format, "L"); EXPECT_STREQ(s.children[1]->format, "z");
    EXPECT_STREQ(s.children[0]->name, "id");
    EXPECT_EQ(static_cast<const std::uint64_t*>(a.children[0]->buffers[1])[0], UINT64_MAX);
    owner_release(o);
    s.release(&s);
    EXPECT_EQ(static_cast<const unsigned char*>(a.children[1]->buffers[2])[1], 255);
    a.release(&a);
}

TEST(BoltArrowOwner, MovedChildrenOutliveAllParentsAndCallerOwner) {
    auto* o = fixture(); ASSERT_NE(o, nullptr);
    ArrowSchema s{}; ArrowArray a{};
    ASSERT_EQ(owner_export_array(o, &s, &a), OwnerStatus::Ok);
    ArrowSchema child_s = *s.children[1]; s.children[1]->release = nullptr;
    ArrowArray child_a = *a.children[1]; a.children[1]->release = nullptr;
    owner_release(o); s.release(&s); a.release(&a);
    EXPECT_STREQ(child_s.format, "z"); EXPECT_STREQ(child_s.name, "raw");
    EXPECT_EQ(static_cast<const unsigned char*>(child_a.buffers[2])[1], 255);
    child_s.release(&child_s);
    EXPECT_EQ(static_cast<const std::int32_t*>(child_a.buffers[1])[2], 3);
    child_a.release(&child_a);
}

TEST(BoltArrowOwner, BoundedLeasesRefuseWithoutOutputMutationAndReuse) {
    auto* o = fixture(1); ASSERT_NE(o, nullptr);
    ArrowSchema s{}, refused{}; refused.flags = 123;
    ArrowArray a{};
    ASSERT_EQ(owner_export_array(o, &s, &a), OwnerStatus::Ok);
    EXPECT_EQ(owner_export_schema(o, &refused), OwnerStatus::Capacity);
    EXPECT_EQ(refused.flags, 123);
    s.release(&s);
    EXPECT_EQ(owner_export_schema(o, &refused), OwnerStatus::Capacity);
    a.release(&a);
    ASSERT_EQ(owner_export_schema(o, &refused), OwnerStatus::Ok);
    refused.release(&refused); owner_release(o);
}

TEST(BoltArrowOwner, StreamLeaseFailureRetriesSameBatchAndExportsOutliveStream) {
    auto* o = fixture(1); ASSERT_NE(o, nullptr);
    ArrowArrayStream stream{}; ASSERT_EQ(owner_export_stream(o, &stream), OwnerStatus::Ok);
    ArrowSchema s{}; ASSERT_EQ(stream.get_schema(&stream, &s), 0);
    ArrowArray a{}; a.length = 77;
    EXPECT_EQ(stream.get_next(&stream, &a), ENOMEM); EXPECT_EQ(a.length, 77);
    EXPECT_NE(stream.get_last_error(&stream), nullptr);
    s.release(&s);
    ASSERT_EQ(stream.get_next(&stream, &a), 0); EXPECT_EQ(a.length, 2);
    ArrowArray eof{}; eof.length = 99;
    EXPECT_EQ(stream.get_next(&stream, &eof), 0); EXPECT_EQ(eof.release, nullptr);
    stream.release(&stream); owner_release(o);
    EXPECT_EQ(static_cast<const std::uint64_t*>(a.children[0]->buffers[1])[1], UINT64_C(9223372036854775808));
    a.release(&a);
}

TEST(BoltArrowOwner, StreamSchemaSurvivesStreamAndEmptyBatchKeepsTypes) {
    auto* o = fixture(3, 0); ASSERT_NE(o, nullptr);
    ArrowArrayStream stream{}; ASSERT_EQ(owner_export_stream(o, &stream), OwnerStatus::Ok);
    ArrowSchema s{}; ArrowArray a{};
    ASSERT_EQ(stream.get_schema(&stream, &s), 0);
    ASSERT_EQ(stream.get_next(&stream, &a), 0);
    stream.release(&stream); owner_release(o);
    EXPECT_EQ(a.length, 0); EXPECT_EQ(s.n_children, 2); EXPECT_STREQ(s.children[0]->format, "L");
    a.release(&a); EXPECT_STREQ(s.children[1]->name, "raw"); s.release(&s);
}

TEST(BoltArrowOwner, InvalidAdmissionAndBorrowedBufferRefuse) {
    ArrowOwner* sentinel = reinterpret_cast<ArrowOwner*>(std::uintptr_t{1});
    EXPECT_EQ(owner_create({1024, 0, 1}, &sentinel), OwnerStatus::Invalid);
    EXPECT_EQ(sentinel, reinterpret_cast<ArrowOwner*>(std::uintptr_t{1}));
    EXPECT_EQ(owner_create({32768, 4096, 64}, &sentinel), OwnerStatus::Capacity);
    ArrowOwner* o = nullptr; ASSERT_EQ(owner_create({65536, 1, 1}, &o), OwnerStatus::Ok);
    std::uint64_t borrowed = 3;
    const void* buffers[]{nullptr, &borrowed}; const void* root_buf[]{nullptr};
    ArrowSchema leaf{}; leaf.format = "L";
    ArrowSchema* schemas[]{&leaf}; ArrowSchema s{};
    s.format = "+s"; s.n_children = 1; s.children = schemas;
    ArrowArray child{}; child.n_buffers = 2; child.buffers = buffers;
    ArrowArray* children[]{&child}; ArrowArray a{};
    a.n_buffers = 1; a.buffers = root_buf; a.n_children = 1; a.children = children;
    EXPECT_EQ(owner_publish(o, s, a), OwnerStatus::Invalid);
    EXPECT_EQ(owner_export_schema(o, &s), OwnerStatus::Invalid);
    owner_release(o);
}

TEST(BoltArrowOwner, ExplicitUnsupportedMetadataAndAliasedOutputRefuse) {
    ArrowOwner* o = nullptr; ASSERT_EQ(owner_create({65536, 1, 1}, &o), OwnerStatus::Ok);
    auto* arena = owner_build_arena(o);
    auto* value = arena->allocate_array<std::int64_t>(1); ASSERT_NE(value, nullptr); *value = -7;
    auto* aliased = arena->allocate_array<ArrowArray>(1); ASSERT_NE(aliased, nullptr); *aliased = {};
    const void* buffers[]{nullptr, value}; const void* root_buffers[]{nullptr};
    ArrowSchema leaf{}; leaf.format = "l"; leaf.metadata = "not admitted";
    ArrowSchema* schemas[]{&leaf}; ArrowSchema root{};
    root.format = "+s"; root.n_children = 1; root.children = schemas;
    ArrowArray child{}; child.length = 1; child.n_buffers = 2; child.buffers = buffers;
    ArrowArray* children[]{&child}; ArrowArray array{};
    array.length = 1; array.n_buffers = 1; array.buffers = root_buffers;
    array.n_children = 1; array.children = children;
    EXPECT_EQ(owner_publish(o, root, array), OwnerStatus::Unsupported);
    leaf.metadata = nullptr;
    ASSERT_EQ(owner_publish(o, root, array), OwnerStatus::Ok);
    ArrowSchema s{};
    EXPECT_EQ(owner_export_array(o, &s, aliased), OwnerStatus::Invalid);
    EXPECT_EQ(s.release, nullptr); EXPECT_EQ(aliased->release, nullptr);
    ArrowArrayStream stream{}; ASSERT_EQ(owner_export_stream(o, &stream), OwnerStatus::Ok);
    EXPECT_EQ(stream.get_next(&stream, aliased), EINVAL);
    EXPECT_NE(stream.get_last_error(&stream), nullptr);
    ArrowArray good{}; ASSERT_EQ(stream.get_next(&stream, &good), 0);
    EXPECT_EQ(static_cast<const std::int64_t*>(good.children[0]->buffers[1])[0], -7);
    good.release(&good); stream.release(&stream); owner_release(o);
}

TEST(BoltArrowOwner, ConcurrentIndependentLeasesMoveReleaseAndReuse) {
    auto* o = fixture(4); ASSERT_NE(o, nullptr);
    alignas(64) std::atomic<bool> valid{true};
    auto worker = [o, &valid](unsigned index) {
        for (unsigned iteration = 0; iteration < 256; ++iteration) {
            ArrowSchema s{}; ArrowArray a{};
            if (owner_export_array(o, &s, &a) != OwnerStatus::Ok) {
                valid.store(false); return;
            }
            ArrowArray moved = *a.children[1]; a.children[1]->release = nullptr;
            if (((iteration + index) & 1) != 0) { a.release(&a); s.release(&s); }
            else { s.release(&s); a.release(&a); }
            if (moved.length != 2 || static_cast<const unsigned char*>(moved.buffers[2])[1] != 255)
                valid.store(false);
            moved.release(&moved);
        }
    };
    std::thread t0(worker, 0), t1(worker, 1), t2(worker, 2), t3(worker, 3);
    t0.join(); t1.join(); t2.join(); t3.join();
    EXPECT_TRUE(valid.load()); owner_release(o);
}

TEST(BoltArrowOwner, LayoutAdmissionRejectsMalformedBuffersAndUnsupportedFormats) {
    ArrowOwner* o = nullptr; ASSERT_EQ(owner_create({65536, 1, 1}, &o), OwnerStatus::Ok);
    auto* arena = owner_build_arena(o);
    auto* values = arena->allocate_array<std::int64_t>(2); ASSERT_NE(values, nullptr);
    const void* buffers[]{nullptr, values, values}; const void* root_buffers[]{nullptr};
    ArrowSchema leaf{}; leaf.format = "l";
    ArrowSchema* schemas[]{&leaf}; ArrowSchema schema{};
    schema.format = "+s"; schema.n_children = 1; schema.children = schemas;
    ArrowArray child{}; child.length = 2; child.n_buffers = 1; child.buffers = buffers;
    ArrowArray* children[]{&child}; ArrowArray array{};
    array.length = 2; array.n_buffers = 1; array.buffers = root_buffers;
    array.n_children = 1; array.children = children;
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Invalid);
    child.n_buffers = 2; buffers[1] = nullptr;
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Invalid);
    buffers[1] = values; child.null_count = 1;
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Invalid);
    child.null_count = 0; leaf.format = "+l";
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Unsupported);
    leaf.format = "z"; child.n_buffers = 3;
    auto* offsets = arena->allocate_array<std::int32_t>(3); ASSERT_NE(offsets, nullptr);
    offsets[0] = 0; offsets[1] = 2; offsets[2] = 1; buffers[1] = offsets;
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Invalid);
    offsets[2] = 3; buffers[2] = nullptr;
    EXPECT_EQ(owner_publish(o, schema, array), OwnerStatus::Invalid);
    buffers[2] = values;
    ASSERT_EQ(owner_publish(o, schema, array), OwnerStatus::Ok);
    owner_release(o);
}

TEST(BoltArrowOwner, ConcurrentSchemaAndArrayRootsReleaseOneLeaseSafely) {
    auto* o = fixture(1); ASSERT_NE(o, nullptr);
    for (unsigned i = 0; i < 64; ++i) {
        ArrowSchema s{}; ArrowArray a{};
        ASSERT_EQ(owner_export_array(o, &s, &a), OwnerStatus::Ok);
        std::thread schema([&s] { s.release(&s); });
        std::thread array([&a] { a.release(&a); });
        schema.join(); array.join();
    }
    owner_release(o);
}

TEST(BoltArrowOwner, EveryTemporalUnitPublishesAndKeepsExactTickBitsIncludingEmpty) {
    const char* formats[]{"tss:", "tsm:", "tsu:", "tsn:", "tDs", "tDm", "tDu", "tDn", "tdm"};
    for (const char* format : formats) {
        for (std::int64_t rows = 0; rows <= 2; rows += 2) {
            SCOPED_TRACE(format);
            SCOPED_TRACE(rows);
            ArrowOwner* o = nullptr;
            ASSERT_EQ(owner_create({65536, 1, 1}, &o), OwnerStatus::Ok);
            auto* ticks = owner_build_arena(o)->allocate_array<std::int64_t>(2);
            ASSERT_NE(ticks, nullptr);
            ticks[0] = INT64_MIN; ticks[1] = INT64_MAX;
            const void* root_buffers[]{nullptr}; const void* buffers[]{nullptr, ticks};
            ArrowSchema leaf{}; leaf.format = format; leaf.name = "time";
            ArrowSchema* schemas[]{&leaf}; ArrowSchema schema{};
            schema.format = "+s"; schema.n_children = 1; schema.children = schemas;
            ArrowArray child{}; child.length = rows; child.n_buffers = 2; child.buffers = buffers;
            ArrowArray* children[]{&child}; ArrowArray array{};
            array.length = rows; array.n_buffers = 1; array.buffers = root_buffers;
            array.n_children = 1; array.children = children;
            ASSERT_EQ(owner_publish(o, schema, array), OwnerStatus::Ok);
            ArrowSchema exported_schema{}; ArrowArray exported_array{};
            ASSERT_EQ(owner_export_array(o, &exported_schema, &exported_array), OwnerStatus::Ok);
            owner_release(o);
            EXPECT_STREQ(exported_schema.children[0]->format, format);
            EXPECT_EQ(exported_array.length, rows);
            if (rows != 0) {
                const auto* values = static_cast<const std::int64_t*>(exported_array.children[0]->buffers[1]);
                EXPECT_EQ(values[0], INT64_MIN); EXPECT_EQ(values[1], INT64_MAX);
            }
            exported_array.release(&exported_array); exported_schema.release(&exported_schema);
        }
    }
}
