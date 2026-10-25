// Object-graph admission fixtures are independent JSON plus external Avro goldens.
#include "bolt/lakehouse/iceberg/object_graph.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/iceberg/manifest_avro.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/lakehouse/iceberg/pinned_read.h"
#include "bolt/lakehouse/object_store/s3_uri.h"
#include "bolt/crypto/sigv4.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

namespace {
using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;
using Status=GraphAdmissionStatus;
constexpr char kRoot[]="s3://graph-bucket/root%20space";
constexpr char kTable[]="s3://graph-bucket/root%20space/tables/orders";
constexpr char kUuid[]="12345678-1234-1234-1234-123456789abc";
std::string uri(const std::string& key) { return std::string(kRoot)+"/"+key; }
std::string metadata_json(int64_t id, bool empty=false, bool duplicate=false) {
    const std::string snapshot="{\"snapshot-id\":"+std::to_string(id)+
        ",\"timestamp-ms\":1000,\"sequence-number\":1,\"schema-id\":0,\"manifest-list\":\""+
        uri("outside/list.json")+"\"}";
    return std::string("{\"format-version\":2,\"table-uuid\":\"")+kUuid+
        "\",\"location\":\""+kTable+"\",\"last-updated-ms\":1000,\"last-sequence-number\":1,"+
        (empty ? "" : "\"current-snapshot-id\":"+std::to_string(id)+",")+
        "\"current-schema-id\":0,\"schemas\":[{\"type\":\"struct\",\"schema-id\":0,\"fields\":["
        "{\"id\":1,\"name\":\"value\",\"required\":true,\"type\":\"long\"}]}],"
        "\"default-spec-id\":0,\"partition-specs\":[{\"spec-id\":0,\"fields\":[]}],"
        "\"default-sort-order-id\":0,\"sort-orders\":[{\"order-id\":0,\"fields\":[]}],\"snapshots\":["+
        (empty ? "" : snapshot+(duplicate ? ","+snapshot : ""))+"]}";
}
std::string file_entry(const std::string& path, uint64_t size, int content=0, int status=1) {
    return "{\"status\":"+std::to_string(status)+",\"snapshot_id\":-1,\"data_file\":{\"content\":"+
        std::to_string(content)+",\"file_path\":\""+path+"\",\"file_format\":\"PARQUET\","
        "\"record_count\":1,\"file_size_in_bytes\":"+std::to_string(size)+",\"partition\":[]}}";
}
std::string list_entry(const std::string& path, uint64_t size, int content=0) {
    return "{\"manifest_path\":\""+path+"\",\"manifest_length\":"+std::to_string(size)+
        ",\"partition_spec_id\":0,\"content\":"+std::to_string(content)+"}";
}
struct Stored {
    std::string key,bytes;
    ObjectVersion version{};
    uint32_t heads=0,ranges=0;
    bool body_allowed=true;
};
struct Origin {
    std::vector<Stored> objects;
    uint32_t forbidden=0,head_count=0,range_count=0,largest=0;
    std::string fail_head,fail_read;
    int head_failure=kOsNotFound,read_failure=kOsPreconditionFailed;
    uint32_t read_fail_at=1;
    bool short_read=false,weak_head=false;
    Stored* find(const char* key) {
        for (auto& object:objects) if (object.key==key) return &object;
        return nullptr;
    }
    Stored& set(const std::string& key, const std::string& bytes, bool body_allowed=true) {
        Stored* item=find(key.c_str());
        if (!item) { objects.push_back({}); item=&objects.back(); }
        item->key=key; item->bytes=bytes; item->body_allowed=body_allowed;
        item->version={}; item->version.size=bytes.size(); std::strcpy(item->version.etag,"\"version-one\"");
        return *item;
    }
    static Origin& self(void* ctx) { return *static_cast<Origin*>(ctx); }
    static int get(void* ctx,const char*,Arena*,const uint8_t**,uint64_t*) noexcept {
        ++self(ctx).forbidden; return kOsIoError;
    }
    static int put(void* ctx,const char*,const uint8_t*,uint64_t) noexcept {
        ++self(ctx).forbidden; return kOsIoError;
    }
    static int list(void* ctx,const char*,ObjectEntry*,uint32_t,uint32_t*) noexcept {
        ++self(ctx).forbidden; return kOsIoError;
    }
    static int remove(void* ctx,const char*) noexcept { ++self(ctx).forbidden; return kOsIoError; }
    static int head(void* ctx,const char*,ObjectMeta*) noexcept { ++self(ctx).forbidden; return kOsIoError; }
    static int range(void* ctx,const char*,uint64_t,uint64_t,uint8_t*,uint64_t*) noexcept {
        ++self(ctx).forbidden; return kOsIoError;
    }
    static int version(void* ctx,const char* key,Arena* io,ObjectVersion* out) noexcept {
        auto& s=self(ctx); ++s.head_count;
        if (s.fail_head==key) return s.head_failure;
        auto* item=s.find(key); if (!item) return kOsNotFound;
        ++item->heads;
        if (!io->allocate(73,1)) return kOsBufferSmall;
        *out=item->version;
        if (s.weak_head) std::strcpy(out->etag,"W/\"weak\"");
        return kOsOk;
    }
    static int conditional(void* ctx,const char* key,const ObjectVersion* expected,
        uint64_t offset,uint32_t length,Arena* io,uint8_t* dst,uint32_t* got) noexcept {
        auto& s=self(ctx); ++s.range_count; s.largest=std::max(s.largest,length); *got=0;
        auto* item=s.find(key); if (!item) return kOsNotFound;
        ++item->ranges;
        if (!item->body_allowed) { ++s.forbidden; return kOsIoError; }
        if (s.fail_read==key && item->ranges==s.read_fail_at) {
            if (!s.short_read) return s.read_failure;
            if (length==0) return kOsBadArg;
            std::memcpy(dst,item->bytes.data()+offset,length-1); *got=length-1; return kOsOk;
        }
        if (expected->size!=item->version.size || std::strcmp(expected->etag,item->version.etag)!=0)
            return kOsPreconditionFailed;
        if (offset>item->bytes.size() || length>item->bytes.size()-offset || length>(1u<<20)) return kOsBadArg;
        if (!io->allocate(83,1)) return kOsBufferSmall;
        std::memcpy(dst,item->bytes.data()+offset,length); *got=length; return kOsOk;
    }
};
class IcebergObjectGraph : public ::testing::Test {
protected:
    Origin origin;
    ObjectStoreVT vt{Origin::get,Origin::put,Origin::list,Origin::remove,Origin::head,
        Origin::put,Origin::range,Origin::version,Origin::conditional};
    ObjectStore store{&vt,&origin};
    Arena metadata{{8u<<20,8u<<20,64,false}},list{{8u<<20,8u<<20,64,false}};
    Arena manifest{{8u<<20,8u<<20,64,false}},io{{8u<<20,8u<<20,64,false}};
    Arena index{{2u<<20,2u<<20,64,false}};
    std::array<PinnedObject,32> directory{};
    GraphAdmissionRequest request{};
    GraphAdmissionLimits limits{16,8,32,2u<<20,8u<<20,37};
    GraphAdmissionWork work{&metadata,&list,&manifest,&io,&index,directory.data(),32};
    GraphAdmissionResult result{};
    std::array<uint8_t,sizeof(result)> before{};
    std::string retained;
    void refresh(int64_t id=-1, bool empty=false, bool duplicate=false) {
        retained=metadata_json(id,empty,duplicate);
        request={}; request.metadata_limits={2u<<20,37}; request.metadata_json=
            reinterpret_cast<const uint8_t*>(retained.data()); request.metadata_length=retained.size();
        request.selection=empty ? SnapshotSelection::Empty : SnapshotSelection::Exact; request.snapshot_id=empty ? 0 : id;
        auto& object=request.object;
        std::strcpy(object.store_location,kRoot); std::strcpy(object.table_location,kTable);
        std::strcpy(object.table_uuid,kUuid); std::strcpy(object.object_key,"metadata/retained.json");
        std::strcpy(object.metadata_uri,uri(object.object_key).c_str());
        object.version.size=retained.size(); std::strcpy(object.version.etag,"\"retained-version\"");
        crypto::sha256_hex(request.metadata_json,request.metadata_length,request.sha256);
    }
    void rebuild_lists(bool duplicate=false) {
        const auto* data=origin.find("outside/data.json"); const auto* deletes=origin.find("outside/delete.json");
        const std::string a=list_entry(uri(data->key),data->bytes.size());
        const std::string b=list_entry(uri(deletes->key),deletes->bytes.size(),1);
        origin.set("outside/list.json","["+a+","+b+(duplicate ? ","+a : "")+"]");
    }
    void SetUp() override {
        origin.objects.reserve(16);
        origin.set("outside/data.parquet",std::string(7,'d'),false);
        origin.set("deletes/position.parquet",std::string(9,'p'),false);
        origin.set("deletes/equality.parquet",std::string(11,'e'),false);
        origin.set("outside/data.json","["+file_entry(uri("outside/data.parquet"),7)+","+
            file_entry(uri("outside/data.parquet"),7,0,0)+","+
            file_entry(uri("never/deleted.parquet"),100,0,2)+"]");
        origin.set("outside/delete.json","["+file_entry(uri("deletes/position.parquet"),9,1)+","+
            file_entry(uri("deletes/equality.parquet"),11,2)+"]");
        rebuild_lists(); refresh();
        for (Arena* arena:{&metadata,&list,&manifest,&io,&index}) arena->set_growth_enabled(false);
        result.snapshot_id=123456; result.object_count=123; std::strcpy(result.metadata.sha256,"sentinel");
        std::memcpy(before.data(),&result,sizeof(result));
    }
    Status admit() { return iceberg_admit_object_graph(&store,&request,&limits,&work,&result); }
    void unchanged() { EXPECT_EQ(std::memcmp(before.data(),&result,sizeof(result)),0); }
    void no_forbidden() { EXPECT_EQ(origin.forbidden,0u); }
    void no_io() { EXPECT_EQ(origin.head_count,0u); EXPECT_EQ(origin.range_count,0u); no_forbidden(); }
    void one_manifest(const std::string& bytes, int content=0) {
        origin.set("outside/data.json",bytes);
        origin.set("outside/list.json","["+list_entry(uri("outside/data.json"),bytes.size(),content)+"]");
    }
};

TEST_F(IcebergObjectGraph, CompleteSortedClosurePinsAllLiveRolesAndDeduplicates) {
    rebuild_lists(true);
    ASSERT_EQ(admit(),Status::Ok); ASSERT_EQ(result.object_count,7u);
    EXPECT_EQ(result.snapshot_id,-1); EXPECT_EQ(result.selection,SnapshotSelection::Exact);
    EXPECT_EQ(std::memcmp(&result.metadata.object,&request.object,sizeof(request.object)),0);
    EXPECT_STREQ(result.metadata.sha256,request.sha256);
    uint64_t bytes=retained.size();
    for (const auto& object:origin.objects) {
        EXPECT_EQ(object.heads,1u) << object.key;
        if (object.body_allowed) { bytes+=object.bytes.size(); EXPECT_GT(object.ranges,0u); }
        else EXPECT_EQ(object.ranges,0u);
    }
    EXPECT_EQ(result.total_metadata_bytes,bytes); EXPECT_LE(origin.largest,limits.request_bytes);
    for (uint32_t i=1;i<result.object_count;++i) EXPECT_LT(std::strcmp(directory[i-1].key,directory[i].key),0);
    no_forbidden();
}
TEST_F(IcebergObjectGraph, EmptyPublishesOnlyCopiedMetadataAndUsesNoTransportCapability) {
    refresh(0,true); vt.head_versioned=nullptr; vt.get_range_if_match=nullptr;
    ASSERT_EQ(admit(),Status::Ok); ASSERT_EQ(result.object_count,1u);
    EXPECT_STREQ(directory[0].key,"metadata/retained.json"); EXPECT_EQ(result.total_metadata_bytes,retained.size());
    no_io(); metadata.reset(); list.reset(); manifest.reset(); io.reset(); index.reset();
    EXPECT_STREQ(result.metadata.object.table_uuid,kUuid); EXPECT_STREQ(result.metadata.sha256,request.sha256);
}
TEST_F(IcebergObjectGraph, LiteralSignedSelectionIncludesBothInt64BoundariesAndZero) {
    for (int64_t id:{INT64_MIN,int64_t{-1},int64_t{0},int64_t{1},INT64_MAX}) {
        refresh(id); ASSERT_EQ(admit(),Status::Ok) << id; EXPECT_EQ(result.snapshot_id,id);
    }
    no_forbidden();
}
TEST_F(IcebergObjectGraph, MissingAndDuplicateSelectedIdsRefuseWithoutOriginIo) {
    request.snapshot_id=24; EXPECT_EQ(admit(),Status::NotFound); unchanged(); no_io();
    refresh(-1,false,true); EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_io();
}
TEST_F(IcebergObjectGraph, EmptyNeedsZeroIdAndAbsentCurrentAndLegacyRefuses) {
    request.selection=SnapshotSelection::Empty; request.snapshot_id=0;
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_io();
    request.snapshot_id=-1; EXPECT_EQ(admit(),Status::BadArg); unchanged(); no_io();
    request.selection=SnapshotSelection::Legacy; EXPECT_EQ(admit(),Status::BadArg); unchanged(); no_io();
}
TEST_F(IcebergObjectGraph, CompleteDigestAndRetainedExtentBindTheSuppliedIdentity) {
    request.sha256[0]=request.sha256[0]=='0' ? '1' : '0';
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_io();
    refresh(); ++request.object.version.size;
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_io();
    refresh(); request.sha256[0]='A'; EXPECT_EQ(admit(),Status::BadArg); unchanged(); no_io();
}
TEST_F(IcebergObjectGraph, UriEscapesAndOutsideReferencesRefuseWithoutRebasing) {
    for (const char* reference:{"relative.parquet","s3://another-bucket/root%20space/data.parquet",
         "s3://graph-bucket/root%20spaces/data.parquet","s3://graph-bucket/root%20space/../data.parquet",
         "s3://graph-bucket/root%20space/a?b.parquet","s3://graph-bucket/root%20space/a#b.parquet"}) {
        one_manifest("["+file_entry(reference,7)+"]");
        const auto rc=admit(); EXPECT_TRUE(rc==Status::Conflict || rc==Status::Unsupported) << reference;
        unchanged();
    }
    EXPECT_EQ(origin.find("outside/data.parquet")->heads,0u); no_forbidden();
}
TEST_F(IcebergObjectGraph, MetadataKeyCannotBeReusedForListManifestOrLiveData) {
    std::string changed=retained; const auto at=changed.find(uri("outside/list.json"));
    changed.replace(at,uri("outside/list.json").size(),request.object.metadata_uri);
    retained=changed; request.metadata_json=reinterpret_cast<const uint8_t*>(retained.data());
    request.metadata_length=retained.size(); request.object.version.size=retained.size();
    crypto::sha256_hex(request.metadata_json,request.metadata_length,request.sha256);
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_io();
    refresh(); origin.set("outside/list.json","["+list_entry(request.object.metadata_uri,retained.size())+"]");
    EXPECT_EQ(admit(),Status::Conflict); unchanged();
    one_manifest("["+file_entry(request.object.metadata_uri,retained.size())+"]");
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, SameKeyRoleAndRepeatedExtentConflictsRefuse) {
    one_manifest("["+file_entry(uri("outside/data.parquet"),7)+","+
        file_entry(uri("outside/data.parquet"),8)+"]");
    EXPECT_EQ(admit(),Status::Conflict); EXPECT_EQ(origin.find("outside/data.parquet")->heads,1u); unchanged();
    origin.find("outside/data.parquet")->heads=0;
    one_manifest("["+file_entry(uri("outside/data.parquet"),7,1)+","+
        file_entry(uri("outside/data.parquet"),7,2)+"]",1);
    EXPECT_EQ(admit(),Status::Conflict); EXPECT_EQ(origin.find("outside/data.parquet")->heads,1u); unchanged();
}
TEST_F(IcebergObjectGraph, ManifestClassAndDeclaredExactLengthsAreRequired) {
    one_manifest("["+file_entry(uri("outside/data.parquet"),7,1)+"]");
    EXPECT_EQ(admit(),Status::Conflict); unchanged();
    one_manifest("["+file_entry(uri("outside/data.parquet"),0)+"]");
    EXPECT_EQ(admit(),Status::Conflict); unchanged();
    origin.set("outside/list.json","["+list_entry(uri("outside/data.json"),0)+"]");
    EXPECT_EQ(admit(),Status::Conflict); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, UnknownEnumsAndMalformedGraphFieldsNeverBecomeDefaults) {
    for (const std::string& body:{
        "["+file_entry(uri("outside/data.parquet"),7,256)+"]",
        "["+file_entry(uri("outside/data.parquet"),7,0,256)+"]",
        std::string("[{\"status\":1,\"data_file\":{\"file_path\":\"")+uri("outside/data.parquet")+"\"}}]",
        std::string("null"),std::string("{}")}) {
        one_manifest(body); EXPECT_NE(admit(),Status::Ok); unchanged();
    }
    one_manifest("[]");
    origin.set("outside/list.json","["+list_entry(uri("outside/data.json"),2,256)+"]");
    EXPECT_EQ(admit(),Status::Unsupported); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, DeletedEntriesDoNotHeadFilesAndUnknownDeletedContentRefuses) {
    one_manifest("["+file_entry(uri("never/deleted.parquet"),123,0,2)+"]");
    ASSERT_EQ(admit(),Status::Ok); EXPECT_EQ(result.object_count,3u); EXPECT_EQ(origin.head_count,2u);
    one_manifest("["+file_entry(uri("never/deleted.parquet"),123,256,2)+"]");
    EXPECT_EQ(admit(),Status::Unsupported); no_forbidden();
}
TEST_F(IcebergObjectGraph, VersionHeadFailuresRemainDistinctAndNeverRepin) {
    for (const auto failure:{std::pair{kOsNotFound,Status::NotFound},
         std::pair{kOsPreconditionFailed,Status::Changed},std::pair{kOsIoError,Status::IoError},
         std::pair{kOsBufferSmall,Status::Capacity}}) {
        origin.fail_head="outside/list.json"; origin.head_failure=failure.first;
        const uint32_t before_heads=origin.head_count;
        EXPECT_EQ(admit(),failure.second); EXPECT_EQ(origin.head_count,before_heads+1); unchanged();
    }
    origin.fail_head.clear(); origin.weak_head=true; EXPECT_EQ(admit(),Status::Unsupported); unchanged();
    no_forbidden();
}
TEST_F(IcebergObjectGraph, ConditionalChangesAndShortSuccessNeverPublishOrRetry) {
    origin.fail_read="outside/list.json"; origin.read_fail_at=2;
    EXPECT_EQ(admit(),Status::Changed); EXPECT_EQ(origin.find("outside/list.json")->ranges,2u); unchanged();
    origin.find("outside/list.json")->ranges=0; origin.short_read=true;
    EXPECT_EQ(admit(),Status::IoError); EXPECT_EQ(origin.find("outside/list.json")->ranges,2u); unchanged();
    no_forbidden();
}
TEST_F(IcebergObjectGraph, FailureInLastFileKeepsResultDescriptorUnchanged) {
    origin.fail_head="deletes/equality.parquet"; origin.head_failure=kOsNotFound;
    EXPECT_EQ(admit(),Status::NotFound); unchanged();
    EXPECT_EQ(origin.find("outside/data.parquet")->heads,1u);
    EXPECT_EQ(origin.find("deletes/position.parquet")->heads,1u); no_forbidden();
}
TEST_F(IcebergObjectGraph, ExactObjectManifestEntryAndByteCapsAdmitAndOneLessRefuses) {
    limits.max_objects=7; limits.max_manifests=2; limits.max_entries=5;
    uint64_t total=retained.size(); for (const auto& object:origin.objects)
        if (object.body_allowed) total+=object.bytes.size();
    limits.max_total_metadata_bytes=total;
    ASSERT_EQ(admit(),Status::Ok); EXPECT_EQ(result.total_metadata_bytes,total);
    std::memcpy(before.data(),&result,sizeof(result));
    --limits.max_objects; EXPECT_EQ(admit(),Status::Capacity); unchanged(); ++limits.max_objects;
    --limits.max_manifests; EXPECT_EQ(admit(),Status::Capacity); unchanged(); ++limits.max_manifests;
    --limits.max_entries; EXPECT_EQ(admit(),Status::Capacity); unchanged(); ++limits.max_entries;
    --limits.max_total_metadata_bytes; EXPECT_EQ(admit(),Status::Capacity); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, FrozenArenaExhaustionReturnsCapacityAndNeverGrowsBacking) {
    Arena tiny{{128,128,64,false}}; tiny.set_growth_enabled(false);
    work.index=&tiny; EXPECT_EQ(admit(),Status::Capacity); unchanged(); no_io();
    work.index=&index; work.manifest=&tiny; EXPECT_EQ(admit(),Status::Capacity); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, InvalidLimitsAndArenaModesRejectBeforeResetAndIo) {
    ASSERT_NE(metadata.allocate(31),nullptr); const auto used=metadata.total_allocated();
    limits.request_bytes=0; EXPECT_EQ(admit(),Status::BadArg); EXPECT_EQ(metadata.total_allocated(),used);
    limits.request_bytes=(1u<<20)+1; EXPECT_EQ(admit(),Status::BadArg);
    limits.request_bytes=37; limits.max_objects=65537; EXPECT_EQ(admit(),Status::BadArg);
    limits.max_objects=16; limits.max_manifests=65536; EXPECT_EQ(admit(),Status::BadArg);
    limits.max_manifests=8; request.metadata_limits.max_bytes=0; EXPECT_EQ(admit(),Status::BadArg);
    request.metadata_limits.max_bytes=2u<<20; io.set_concurrent(true); EXPECT_EQ(admit(),Status::BadArg);
    io.set_concurrent(false); manifest.set_growth_enabled(true); EXPECT_EQ(admit(),Status::BadArg);
    unchanged(); no_io();
}

TEST_F(IcebergObjectGraph, AliasesOfControlsDirectoryResultsAndVtablesRejectBeforeArenaInspection) {
    ASSERT_NE(metadata.allocate(31),nullptr); const auto used=metadata.total_allocated();
    Arena* saved=work.metadata;
    for (void* bad:{static_cast<void*>(&request),static_cast<void*>(&limits),static_cast<void*>(&work),
                   static_cast<void*>(&result),static_cast<void*>(&store),static_cast<void*>(&vt),
                   static_cast<void*>(directory.data())}) {
        work.metadata=reinterpret_cast<Arena*>(bad);
        EXPECT_EQ(admit(),Status::BadArg); unchanged();
    }
    work.metadata=saved;
    Arena* saved_io=work.io; work.io=work.list;
    EXPECT_EQ(admit(),Status::BadArg); work.io=saved_io;
    auto* saved_objects=work.objects; work.objects=reinterpret_cast<PinnedObject*>(&request);
    EXPECT_EQ(admit(),Status::BadArg); work.objects=saved_objects;
    EXPECT_EQ(iceberg_admit_object_graph(&store,&request,&limits,&work,
        reinterpret_cast<GraphAdmissionResult*>(&work)),Status::BadArg);
    EXPECT_EQ(metadata.total_allocated(),used); unchanged(); no_io();
}
TEST_F(IcebergObjectGraph, InputsAndOutputDirectoryInAnyArenaBackingRefuseBeforeReset) {
    auto* saved_objects=work.objects;
    for (Arena* arena:{&metadata,&list,&manifest,&io,&index}) {
        auto* memory=arena->allocate_array<PinnedObject>(32); ASSERT_NE(memory,nullptr);
        work.objects=memory; const auto used=arena->total_allocated();
        EXPECT_EQ(admit(),Status::BadArg); EXPECT_EQ(arena->total_allocated(),used); unchanged();
    }
    work.objects=saved_objects;
    auto* bytes=io.allocate_array<uint8_t>(retained.size()); ASSERT_NE(bytes,nullptr);
    std::memcpy(bytes,retained.data(),retained.size()); request.metadata_json=bytes;
    const auto used=io.total_allocated(); EXPECT_EQ(admit(),Status::BadArg);
    EXPECT_EQ(io.total_allocated(),used); unchanged(); no_io();
}
TEST_F(IcebergObjectGraph, LargeActualManifestBodyIsReadInBoundedRequests) {
    std::string bytes="["+file_entry(uri("outside/data.parquet"),7)+"]";
    bytes.append((17u<<20)-bytes.size(),' '); one_manifest(bytes);
    Arena large{{40u<<20,40u<<20,64,false}}; large.set_growth_enabled(false); work.manifest=&large;
    limits.max_metadata_object_bytes=20u<<20; limits.max_total_metadata_bytes=24u<<20;
    limits.request_bytes=1u<<20;
    ASSERT_EQ(admit(),Status::Ok); EXPECT_GT(result.total_metadata_bytes,16u<<20);
    EXPECT_EQ(origin.find("outside/data.json")->ranges,17u); EXPECT_LE(origin.largest,1u<<20);
    EXPECT_EQ(origin.find("outside/data.parquet")->ranges,0u); no_forbidden();
}
TEST_F(IcebergObjectGraph, PositiveOwnedAvroGraphIncludesDataAndBothDeleteClasses) {
    Arena writer;
    constexpr char schema[]=R"({"type":"struct","schema-id":0,"fields":[{"id":1,"name":"value","required":true,"type":"long"}]})";
    std::array<DataFileRef,3> files{};
    const char* paths[]{"outside/data.parquet","deletes/position.parquet","deletes/equality.parquet"};
    const uint64_t sizes[]{7,9,11};
    for (uint32_t i=0;i<3;++i) {
        files[i].status=ManifestStatus::kAdded; files[i].content=static_cast<FileContent>(i);
        std::strcpy(files[i].file_path,uri(paths[i]).c_str()); files[i].stats.file_size_in_bytes=static_cast<int64_t>(sizes[i]);
        files[i].stats.record_count=1;
    }
    files[2].equality_ids[0]=1; files[2].n_equality_ids=1;   // an equality delete names its key (G2ICE-283)
    const uint8_t* bytes=nullptr; uint64_t size=0;
    ASSERT_TRUE(manifest_write_avro(files.data(),1,-1,1,schema,sizeof(schema)-1,0,&writer,&bytes,&size));
    origin.set("outside/data.json",std::string(reinterpret_cast<const char*>(bytes),static_cast<size_t>(size)));
    writer.reset();
    ASSERT_TRUE(manifest_write_avro(files.data()+1,2,-1,1,schema,sizeof(schema)-1,0,&writer,&bytes,&size));
    origin.set("outside/delete.json",std::string(reinterpret_cast<const char*>(bytes),static_cast<size_t>(size)));
    writer.reset(); std::array<ManifestListEntry,2> entries{};
    for (uint32_t i=0;i<2;++i) {
        const auto* object=origin.find(i==0 ? "outside/data.json" : "outside/delete.json");
        ASSERT_LE(object->bytes.size(),static_cast<uint64_t>(INT64_MAX));
        std::strcpy(entries[i].manifest_path,uri(object->key).c_str());
        entries[i].manifest_length=static_cast<int64_t>(object->bytes.size());
        entries[i].content=i==0 ? ManifestContent::kDataManifest : ManifestContent::kDeleteManifest;
        entries[i].added_files_count=i==0 ? 1 : 2; entries[i].added_rows_count=entries[i].added_files_count;
    }
    ASSERT_TRUE(manifest_list_write_avro(entries.data(),2,-1,1,&writer,&bytes,&size));
    origin.set("outside/list.json",std::string(reinterpret_cast<const char*>(bytes),static_cast<size_t>(size)));
    ASSERT_EQ(admit(),Status::Ok); EXPECT_EQ(result.object_count,7u);
    EXPECT_EQ(origin.find("outside/data.parquet")->ranges,0u);
    EXPECT_EQ(origin.find("deletes/position.parquet")->ranges,0u);
    EXPECT_EQ(origin.find("deletes/equality.parquet")->ranges,0u); no_forbidden();
}

TEST_F(IcebergObjectGraph, JsonFloatingDuplicateAndEscapedCriticalFieldsRefuse) {
    const std::string original="["+file_entry(uri("outside/data.parquet"),7)+"]";
    const std::array<std::pair<std::string,std::string>,4> changes{{
        {"\"content\":0","\"content\":0.0"},
        {"\"file_size_in_bytes\":7","\"file_size_in_bytes\":7.0"},
        {"\"content\":0","\"content\":0,\"content\":0"},
        {"\"file_path\"","\"file_\\u0070ath\""}}};
    for (const auto& change:changes) {
        std::string bytes=original; const auto at=bytes.find(change.first); ASSERT_NE(at,std::string::npos);
        bytes.replace(at,change.first.size(),change.second); one_manifest(bytes);
        EXPECT_EQ(admit(),Status::DecodeFailed); unchanged();
    }
    no_forbidden();
}
TEST_F(IcebergObjectGraph, AvroIntegerContentDoesNotNarrowUnknownValuesIntoValidRoles) {
    using namespace bolt::ingest;
    std::array<AvroField,3> fields{}; std::array<AvroValue,3> values{};
    std::strcpy(fields[0].name,"manifest_path"); fields[0].type=AvroType::kString;
    std::strcpy(fields[1].name,"manifest_length"); fields[1].type=AvroType::kLong;
    std::strcpy(fields[2].name,"content"); fields[2].type=AvroType::kLong;
    const std::string path=uri("outside/data.json");
    values[0].type=AvroType::kString; values[0].bytes=reinterpret_cast<const uint8_t*>(path.data());
    values[0].bytes_len=static_cast<uint32_t>(path.size());
    values[1].type=AvroType::kLong; values[1].num.i64=static_cast<int64_t>(origin.find("outside/data.json")->bytes.size());
    values[2].type=AvroType::kLong; values[2].num.i64=256;
    std::array<uint8_t,8192> buffer{}; uint64_t size=buffer.size(); uint8_t sync[kAvroSyncLen]{};
    ASSERT_TRUE(avro_write(fields.data(),3,values.data(),1,sync,buffer.data(),&size));
    origin.set("outside/list.json",std::string(reinterpret_cast<const char*>(buffer.data()),static_cast<size_t>(size)));
    EXPECT_EQ(admit(),Status::Unsupported); unchanged(); EXPECT_EQ(origin.head_count,1u); no_forbidden();
}

// Both owned writers below emit one block with two rows. Change ONLY that
// block's zig-zag count from two (4) to one (2), retaining the entire encoded
// body, declared block size, header and sync marker. A reader that consumes
// just the claimed row and skips the body remainder would admit a partial graph.
bool lower_two_row_block_count(std::string* bytes) {
    Arena scratch; bolt::ingest::AvroHeader header{}; uint64_t body=0;
    if (!bolt::ingest::avro_read_header(reinterpret_cast<const uint8_t*>(bytes->data()),
        bytes->size(),&scratch,&header,&body) || body>=bytes->size() || (*bytes)[body]!=4) return false;
    (*bytes)[body]=2; return true;
}
TEST_F(IcebergObjectGraph, AvroListBlockMustConsumeAllEncodedRowsBeforePublishingClosure) {
    using namespace bolt::ingest;
    std::array<AvroField,3> fields{}; std::array<AvroValue,6> values{};
    std::strcpy(fields[0].name,"manifest_path"); fields[0].type=AvroType::kString;
    std::strcpy(fields[1].name,"manifest_length"); fields[1].type=AvroType::kLong;
    std::strcpy(fields[2].name,"content"); fields[2].type=AvroType::kLong;
    const std::string path=uri("outside/data.json");
    for (uint32_t row=0;row<2;++row) {
        auto* value=values.data()+row*3;
        value[0].type=AvroType::kString; value[0].bytes=reinterpret_cast<const uint8_t*>(path.data());
        value[0].bytes_len=static_cast<uint32_t>(path.size());
        value[1].type=AvroType::kLong;
        value[1].num.i64=static_cast<int64_t>(origin.find("outside/data.json")->bytes.size());
        value[2].type=AvroType::kLong; value[2].num.i64=0;
    }
    std::array<uint8_t,8192> buffer{}; uint64_t size=buffer.size(); uint8_t sync[kAvroSyncLen]{};
    ASSERT_TRUE(avro_write(fields.data(),3,values.data(),2,sync,buffer.data(),&size));
    std::string bytes(reinterpret_cast<const char*>(buffer.data()),static_cast<size_t>(size));
    ASSERT_TRUE(lower_two_row_block_count(&bytes)); origin.set("outside/list.json",bytes);
    EXPECT_EQ(admit(),Status::DecodeFailed); unchanged(); no_forbidden();
}
TEST_F(IcebergObjectGraph, AvroManifestBlockMustConsumeAllEncodedRowsBeforePublishingClosure) {
    Arena writer;
    constexpr char schema[]=R"({"type":"struct","schema-id":0,"fields":[{"id":1,"name":"value","required":true,"type":"long"}]})";
    std::array<DataFileRef,2> files{};
    for (auto& file:files) {
        file.status=ManifestStatus::kAdded; file.content=FileContent::kData;
        std::strcpy(file.file_path,uri("outside/data.parquet").c_str());
        file.stats.file_size_in_bytes=7; file.stats.record_count=1;
    }
    const uint8_t* encoded=nullptr; uint64_t size=0;
    ASSERT_TRUE(manifest_write_avro(files.data(),2,-1,1,schema,sizeof(schema)-1,0,&writer,&encoded,&size));
    std::string bytes(reinterpret_cast<const char*>(encoded),static_cast<size_t>(size));
    ASSERT_TRUE(lower_two_row_block_count(&bytes)); one_manifest(bytes);
    EXPECT_EQ(admit(),Status::DecodeFailed); unchanged(); no_forbidden();
}

std::string fixture(const char* name) {
    const std::string path=std::string(BOLT_TEST_DATA_DIR)+"/"+name;
    FILE* file=std::fopen(path.c_str(),"rb"); if (!file) return {};
    if (std::fseek(file,0,SEEK_END)!=0) { std::fclose(file); return {}; }
    const long length=std::ftell(file);
    if (length<=0 || std::fseek(file,0,SEEK_SET)!=0) { std::fclose(file); return {}; }
    std::string bytes(static_cast<size_t>(length),'\0');
    const size_t got=std::fread(bytes.data(),1,bytes.size(),file); std::fclose(file);
    return got==bytes.size() ? bytes : std::string{};
}
TEST_F(IcebergObjectGraph, UnchangedExternalAvroGoldenRefusesNonS3ManifestReferences) {
    const std::string bytes=fixture("golden_iceberg_real_manifest_list.avro"); ASSERT_FALSE(bytes.empty());
    origin.set("outside/list.json",bytes);
    EXPECT_EQ(admit(),Status::Unsupported); unchanged(); EXPECT_EQ(origin.head_count,1u); no_forbidden();
}
TEST_F(IcebergObjectGraph, IndependentExternalS3AvroGraphAdmitsWithoutFixtureRewriting) {
    const std::string external_metadata=fixture("golden_iceberg_s3_graph_metadata.json");
    const std::string external_list=fixture("golden_iceberg_s3_graph_manifest_list.avro");
    const std::string external_manifest=fixture("golden_iceberg_s3_graph_manifest.avro");
    ASSERT_FALSE(external_metadata.empty()); ASSERT_FALSE(external_list.empty()); ASSERT_FALSE(external_manifest.empty());
    Arena scratch; Metadata decoded{};
    ASSERT_TRUE(metadata_parse(reinterpret_cast<const uint8_t*>(external_metadata.data()),
        static_cast<uint32_t>(external_metadata.size()),&scratch,&decoded));
    ASSERT_STREQ(decoded.location,kTable); ASSERT_TRUE(decoded.current_snapshot_present);
    const Snapshot* selected=snapshot_by_id(&decoded,decoded.current_snapshot_id); ASSERT_NE(selected,nullptr);
    char list_key[kOsMaxKey]{};
    ASSERT_EQ(s3_uri_object_key(kRoot,selected->manifest_list,list_key,sizeof(list_key)),S3UriStatus::Ok);
    std::array<ManifestListEntry,8> entries{}; uint32_t count=0;
    ASSERT_TRUE(manifest_list_parse_avro(reinterpret_cast<const uint8_t*>(external_list.data()),
        external_list.size(),&scratch,entries.data(),static_cast<uint32_t>(entries.size()),&count));
    ASSERT_EQ(count,1u); EXPECT_EQ(entries[0].manifest_length,external_manifest.size());
    char manifest_key[kOsMaxKey]{};
    ASSERT_EQ(s3_uri_object_key(kRoot,entries[0].manifest_path,manifest_key,sizeof(manifest_key)),S3UriStatus::Ok);
    std::array<DataFileRef,8> files{}; uint32_t nfiles=0;
    ASSERT_TRUE(manifest_parse_avro(reinterpret_cast<const uint8_t*>(external_manifest.data()),external_manifest.size(),
        &scratch,entries[0].partition_spec_id,files.data(),static_cast<uint32_t>(files.size()),&nfiles));
    ASSERT_GT(nfiles,0u); origin.objects.clear();
    for (uint32_t i=0;i<nfiles;++i) {
        if (files[i].status==ManifestStatus::kDeleted) continue;
        char key[kOsMaxKey]{};
        ASSERT_EQ(s3_uri_object_key(kRoot,files[i].file_path,key,sizeof(key)),S3UriStatus::Ok);
        auto& object=origin.set(key,"",false);
        ASSERT_GE(files[i].stats.file_size_in_bytes,0);
        object.version.size=static_cast<uint64_t>(files[i].stats.file_size_in_bytes);
    }
    origin.set(list_key,external_list); origin.set(manifest_key,external_manifest);
    retained=external_metadata; request.metadata_json=reinterpret_cast<const uint8_t*>(retained.data());
    request.metadata_length=retained.size(); request.object.version.size=retained.size();
    std::strcpy(request.object.table_uuid,decoded.table_uuid); request.snapshot_id=decoded.current_snapshot_id;
    crypto::sha256_hex(request.metadata_json,request.metadata_length,request.sha256);
    ASSERT_EQ(admit(),Status::Ok); EXPECT_GE(result.object_count,4u);
    for (const auto& object:origin.objects) { EXPECT_EQ(object.heads,1u); if (!object.body_allowed) EXPECT_EQ(object.ranges,0u); }
    no_forbidden();
}
TEST_F(IcebergObjectGraph, SortedAdmittedDirectoryInitializesSealedSourceAndPinnedOpener) {
    // Empty tables exercise exact metadata pin/hash/selection through the
    // existing strict opener without inventing Parquet data for this fixture.
    refresh(0,true); ASSERT_EQ(admit(),Status::Ok);
    PinnedObjectStore sealed{}; ObjectStore view{};
    PinnedObjectStoreLimits read_limits{16,2u<<20,37}; io.reset();
    ASSERT_EQ(pinned_object_store_init(&sealed,&store,kRoot,result.objects,result.object_count,
        &read_limits,&io,&view),kOsOk);
    Arena owner{{8u<<20,8u<<20,64,false}},parsed{{8u<<20,8u<<20,64,false}};
    owner.set_growth_enabled(false); parsed.set_growth_enabled(false);
    PinnedTableRead read{}; read.object=request.object; read.metadata_limits=request.metadata_limits;
    std::strcpy(read.sha256,request.sha256); read.selection=SnapshotSelection::Empty;
    read.snapshot_id=0; read.parquet_metadata_bytes=1u<<20; read.manifest_bytes=1u<<20;
    TableHandle* table=nullptr;
    ASSERT_TRUE(iceberg_table_open_pinned(&table,&owner,&parsed,&sealed,&read,
        request.metadata_json,request.metadata_length));
    ASSERT_NE(table,nullptr); no_io();
}
} // namespace
