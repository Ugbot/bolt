// Retained metadata admission and exact scans over a sealed immutable graph.
// JSON metadata/list/manifest fixtures are independent of the Iceberg writer.
#include <gtest/gtest.h>
#include "bolt/bolt_column.h"
#include "bolt_iceberg_nested_json.h"
#include <fstream>
#include <cstdlib>
#include "bolt/crypto/sigv4.h"
#include "bolt/ingest/bolt_parquet_read.h"
#include "bolt/ingest/bolt_parquet_write.h"
#include "bolt/ingest/bolt_parquet_nested.h"
#include "bolt/lakehouse/iceberg/pinned_read.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace bolt;
using namespace bolt::lakehouse;
using namespace bolt::lakehouse::iceberg;
namespace pq = bolt::ingest::parquet;
constexpr char kRoot[] = "s3://read-bucket/root space";
constexpr char kTable[] = "s3://read-bucket/root space/tables/orders";
constexpr char kUuid[] = "12345678-1234-1234-1234-123456789abc";
constexpr uint32_t kRequest = 64;
enum Slot : uint32_t { MetadataSlot, ExactList, LatestList, ExactManifest,
                       LatestManifest, ExactData, LatestData, DeleteManifest, DeleteData, SlotCount };
enum class Columns { Valid, MissingId, DuplicateId, NegativeId, UnknownId, WrongType };
enum class DeleteColumns { Valid, MissingPathId, MissingPositionId, DuplicatePathId,
                           DuplicatePositionId, InvalidPathId, InvalidPositionId };

struct Stored {
    std::string key;
    std::vector<uint8_t> bytes;
    ObjectVersion current{};
    uint32_t calls = 0;
};

struct ConditionalOrigin {
    std::array<Stored, SlotCount> objects{};
    ObjectStore store{};
    uint32_t ranges = 0, forbidden = 0, unknown = 0, largest = 0;
    int fail_slot = -1;
    uint32_t fail_call = 0;
    bool short_success = false;

    static ConditionalOrigin& self(void* p) { return *static_cast<ConditionalOrigin*>(p); }
    static int get(void* p, const char*, Arena*, const uint8_t**, uint64_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int put(void* p, const char*, const uint8_t*, uint64_t) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int list(void* p, const char*, ObjectEntry*, uint32_t, uint32_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int remove(void* p, const char*) noexcept { ++self(p).forbidden; return kOsIoError; }
    static int head(void* p, const char*, ObjectMeta*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int version(void* p, const char*, Arena*, ObjectVersion*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int range(void* p, const char*, uint64_t, uint64_t, uint8_t*, uint64_t*) noexcept {
        ++self(p).forbidden; return kOsIoError;
    }
    static int conditional(void* p, const char* key, const ObjectVersion* pin,
        uint64_t offset, uint32_t length, Arena* scratch, uint8_t* dst, uint32_t* got) noexcept {
        auto& s = self(p); ++s.ranges; *got = 0;
        s.largest = std::max(s.largest, length);
        for (uint32_t i = 0; i < SlotCount; ++i) {
            auto& object = s.objects[i];
            if (object.key != key) continue;
            ++object.calls;
            if (static_cast<int>(i) == s.fail_slot && object.calls == s.fail_call) {
                if (!s.short_success) std::strcpy(object.current.etag, "\"changed\"");
                else {
                    if (length == 0) return kOsBadArg;
                    std::memcpy(dst, object.bytes.data() + offset, length - 1u);
                    *got = length - 1u; return kOsOk;
                }
            }
            if (pin->size != object.current.size || std::strcmp(pin->etag, object.current.etag) != 0)
                return kOsPreconditionFailed;
            if (length == 0 || length > kRequest || offset > object.bytes.size() ||
                length > object.bytes.size() - offset) return kOsBadArg;
            auto* response = static_cast<uint8_t*>(scratch->allocate(length, 1));
            if (response == nullptr) return kOsBufferSmall;
            std::memcpy(response, object.bytes.data() + offset, length);
            std::memcpy(dst, response, length); *got = length; return kOsOk;
        }
        ++s.unknown; return kOsNotFound;
    }
    void initialize() {
        static const ObjectStoreVT vt{get, put, list, remove, head, put, range, version, conditional};
        store = {&vt, this};
    }
};

std::string uri(const char* suffix) { return std::string(kRoot) + "/" + suffix; }

std::string metadata_json(int64_t exact_id, bool absent = false) {
    return std::string(R"({"format-version":2,"table-uuid":")") + kUuid +
        R"(","location":")" + kTable + R"(","last-updated-ms":2000,"last-sequence-number":2,)" +
        (absent ? "" : "\"current-snapshot-id\":19,") +
        R"("current-schema-id":9,"schemas":[{"type":"struct","schema-id":9,"fields":[)"
        R"({"id":7,"name":"trade_id","required":true,"type":"long"},)"
        R"({"id":11,"name":"quote","required":true,"type":"double"}]}],)"
        R"("default-spec-id":0,"partition-specs":[{"spec-id":0,"fields":[]}],)"
        R"("default-sort-order-id":0,"sort-orders":[{"order-id":0,"fields":[]}],"snapshots":[)" +
        (absent ? std::string() : std::string("{\"snapshot-id\":") + std::to_string(exact_id) +
        R"(,"timestamp-ms":1000,"sequence-number":1,"schema-id":9,"manifest-list":")" +
        uri("lists/exact.json") + R"("},{"snapshot-id":19,"timestamp-ms":2000,"sequence-number":2,)"
        R"("schema-id":9,"manifest-list":")" + uri("lists/latest.json") + "\"}") + "]}";
}

std::string manifest_list(const std::string& path, int64_t snapshot) {
    return "[{\"manifest_path\":\"" + path +
        "\",\"manifest_length\":0,\"partition_spec_id\":0,\"content\":0,\"added_snapshot_id\":" +
        std::to_string(snapshot) + ",\"added_files_count\":1}]";
}

std::string manifest(const std::string& path, int64_t snapshot, uint64_t size) {
    return "[{\"status\":1,\"snapshot_id\":" + std::to_string(snapshot) +
        ",\"data_file\":{\"content\":0,\"file_path\":\"" + path +
        "\",\"file_format\":\"PARQUET\",\"record_count\":3,\"file_size_in_bytes\":" +
        std::to_string(size) + ",\"partition\":[]}}]";
}

bool parquet_bytes(std::vector<uint8_t>* out, bool latest, Columns mode,bool groups=false) {
    Arena arena;
    BoltBatch batch{}; BoltBatch::init_empty(&batch);
    if (!BoltBatch::alloc_columns(&batch, &arena, 2) ||
        !batch.schema.add_field("legacy_quote", BoltType::Float64, false) ||
        !batch.schema.add_field("legacy_trade", BoltType::Int64, false)) return false;
    batch.num_rows = 3; batch.num_cols = 2; batch.arena = &arena;
    batch.columns[0][0] = BoltColumn::make_flat_alloc(3, BoltType::Float64, &arena);
    batch.columns[0][1] = BoltColumn::make_flat_alloc(3, BoltType::Int64, &arena);
    if (batch.columns[0][0].data == nullptr || batch.columns[0][1].data == nullptr) return false;
    for (uint32_t i = 0; i < 3; ++i) {
        static_cast<double*>(batch.columns[0][0].data)[i] = (i + 1u) * 1.25;
        static_cast<int64_t*>(batch.columns[0][1].data)[i] = (latest ? 901 : 41) + i;
    }
    pq::ParquetWriteOpts opts{}; opts.n_columns = 2; opts.compression = 0;
    std::strcpy(opts.columns[0].name, "legacy_quote");
    std::strcpy(opts.columns[1].name, "legacy_trade");
    opts.columns[0].type = BoltType::Float64; opts.columns[1].type = BoltType::Int64;
    opts.columns[0].has_field_id = opts.columns[1].has_field_id = true;
    opts.columns[0].field_id = 11; opts.columns[1].field_id = 7;
    if (mode == Columns::MissingId) opts.columns[0].has_field_id = false;
    if (mode == Columns::DuplicateId) opts.columns[0].field_id = 7;
    if (mode == Columns::NegativeId) opts.columns[0].field_id = -11;
    if (mode == Columns::UnknownId) opts.columns[0].field_id = 99;
    // Valid Parquet with swapped IDs has incompatible types for both logical fields.
    if (mode == Columns::WrongType) { opts.columns[0].field_id = 7; opts.columns[1].field_id = 11; }
    auto* writer = pq::parquet_write_open_mem(&opts);
    if (writer == nullptr) return false;
    bool wrote=true;
    if (!groups) wrote=pq::parquet_write_row_group(writer,&batch);
    else {
        auto* quotes=static_cast<double*>(batch.columns[0][0].data);
        auto* trades=static_cast<int64_t*>(batch.columns[0][1].data);
        batch.num_rows=1; batch.columns[0][0].length=1; batch.columns[0][1].length=1;
        for (uint32_t i=0; i<3 && wrote; ++i) {
            batch.columns[0][0].data=quotes+i; batch.columns[0][1].data=trades+i;
            wrote=pq::parquet_write_row_group(writer,&batch);
        }
    }
    const uint8_t* bytes = nullptr; uint64_t length = 0;
    const bool closed = pq::parquet_write_close_mem(writer, &arena, &bytes, &length);
    if (!wrote || !closed || bytes == nullptr || length == 0) return false;
    out->assign(bytes, bytes + static_cast<size_t>(length)); return true;
}

// Independent compact-Thrift footer encoder. Explicit field headers avoid
// depending on the production writer's delta-header/schema-tree encoder.
struct FooterBytes {
    std::vector<uint8_t> bytes;
    void var(uint64_t n) {
        for (uint32_t i=0; i<10; ++i) {
            const auto low=static_cast<uint8_t>(n&127u); n>>=7;
            bytes.push_back(static_cast<uint8_t>(low | (n ? 128u : 0u))); if (!n) return;
        }
    }
    void zig(int64_t n) { var((static_cast<uint64_t>(n)<<1)^static_cast<uint64_t>(n>>63)); }
    void field(uint16_t id,uint8_t type) { bytes.push_back(type); zig(id); }
    void number(uint16_t id,int64_t n,uint8_t type=pq::kTcI32) { field(id,type); zig(n); }
    void string(const std::string& value) { var(value.size()); bytes.insert(bytes.end(),value.begin(),value.end()); }
    void name(const char* value) { field(4,pq::kTcBinary); string(value); }
    void list(uint32_t n,uint8_t type) {
        bytes.push_back(static_cast<uint8_t>((n<15 ? n : 15u)<<4 | type)); if (n>=15) var(n);
    }
    void stop() { bytes.push_back(0); }
};

void fixture_schema(FooterBytes* out,const pq::PqMeta& meta,bool nested,bool malformed) {
    auto& b=*out; b.field(2,pq::kTcList); b.list(nested ? 4u : 3u,pq::kTcStruct);
    b.name("schema"); b.number(5,nested ? 1 : 2); b.stop();
    if (nested) { b.number(3,0); b.name("required_group"); b.number(5,2); b.number(9,100); b.stop(); }
    for (uint32_t i=0; i<2; ++i) {
        const auto& column=meta.columns[i]; b.number(1,static_cast<int32_t>(column.physical));
        b.number(3,0); b.name(column.name); b.number(9,column.field_id);
        if (column.logical==static_cast<int32_t>(pq::PqLogical::String) || column.converted==0) b.number(6,0);
        if (malformed && column.physical==pq::PqType::Int64) {
            b.field(10,pq::kTcStruct); b.field(10,pq::kTcStruct);
            b.field(1,pq::kTcByte); b.bytes.push_back(64); // required isSigned member deliberately absent
            b.stop(); b.stop();
        }
        b.stop();
    }
}

void fixture_row_group(FooterBytes* out,const pq::PqMeta& meta,bool nested) {
    auto& b=*out; b.field(4,pq::kTcList); b.list(1,pq::kTcStruct);
    b.field(1,pq::kTcList); b.list(2,pq::kTcStruct);
    int64_t total=0;
    for (uint32_t i=0; i<2; ++i) {
        const auto& chunk=meta.chunks[i]; total+=chunk.total_uncompressed_size;
        b.number(2,chunk.data_page_offset,pq::kTcI64); b.field(3,pq::kTcStruct);
        b.number(1,static_cast<int32_t>(meta.columns[i].physical));
        b.field(2,pq::kTcList); b.list(chunk.dictionary_page_offset>0 ? 3u : 2u,pq::kTcI32);
        b.zig(0); b.zig(3); if (chunk.dictionary_page_offset>0) b.zig(8);
        b.field(3,pq::kTcList); b.list(nested ? 2u : 1u,pq::kTcBinary);
        if (nested) b.string("required_group");
        b.string(meta.columns[i].name);
        b.number(4,static_cast<int32_t>(chunk.codec)); b.number(5,chunk.num_values,pq::kTcI64);
        b.number(6,chunk.total_uncompressed_size,pq::kTcI64);
        b.number(7,chunk.total_compressed_size,pq::kTcI64);
        b.number(9,chunk.data_page_offset,pq::kTcI64);
        if (chunk.dictionary_page_offset>0) b.number(11,chunk.dictionary_page_offset,pq::kTcI64);
        b.stop(); b.stop();
    }
    b.number(2,total,pq::kTcI64); b.number(3,meta.num_rows,pq::kTcI64); b.stop();
}

bool rewrite_footer(std::vector<uint8_t>* file,bool nested=true,bool malformed=false) {
    if (file->size()<12) return false;
    Arena arena; auto* meta=arena.allocate_array<pq::PqMeta>(1);
    if (meta==nullptr) return false;
    std::memset(meta,0,sizeof(*meta));
    if (!pq::parquet_read_meta(file->data(),file->size(),&arena,meta) ||
        meta->n_columns!=2 || meta->n_row_groups!=1) return false;
    FooterBytes b; b.number(1,1); fixture_schema(&b,*meta,nested,malformed);
    b.number(3,meta->num_rows,pq::kTcI64); fixture_row_group(&b,*meta,nested); b.stop();
    const size_t length_at=file->size()-8u; uint32_t old_length=0;
    for (uint32_t i=0; i<4; ++i) old_length |= uint32_t((*file)[length_at+i])<<(8u*i);
    if (old_length>length_at-4u || b.bytes.size()>UINT32_MAX) return false;
    file->resize(length_at-old_length); file->insert(file->end(),b.bytes.begin(),b.bytes.end());
    const auto length=static_cast<uint32_t>(b.bytes.size());
    for (uint32_t i=0; i<4; ++i) file->push_back(static_cast<uint8_t>(length>>(8u*i)));
    for (const char c : std::array<char,4>{'P','A','R','1'}) file->push_back(static_cast<uint8_t>(c));
    return true;
}

bool delete_parquet_bytes(std::vector<uint8_t>* out,bool reordered,bool nul,DeleteColumns mode) {
    Arena arena; BoltBatch batch{}; BoltBatch::init_empty(&batch);
    if (!BoltBatch::alloc_columns(&batch,&arena,2)) return false;
    const uint32_t path=reordered ? 1u : 0u, pos=1u-path;
    const char* names[2]{reordered ? "old_position" : "file_path",reordered ? "old_path" : "pos"};
    const BoltType types[2]{reordered ? BoltType::Int64 : BoltType::Utf8,
                            reordered ? BoltType::Utf8 : BoltType::Int64};
    for (uint32_t i=0; i<2; ++i) if (!batch.schema.add_field(names[i],types[i],false)) return false;
    std::string path_bytes=uri("shared/exact%25part.parquet");
    if (nul) { path_bytes.push_back('\0'); path_bytes+="untrusted-tail"; }
    const int32_t offsets[2]{0,static_cast<int32_t>(path_bytes.size())};
    batch.columns[0][path]=BoltColumn::make_utf8_from_packed(path_bytes.data(),nullptr,offsets,1,&arena);
    batch.columns[0][pos]=BoltColumn::make_flat_alloc(1,BoltType::Int64,&arena);
    if (batch.columns[0][path].data==nullptr || batch.columns[0][pos].data==nullptr) return false;
    static_cast<int64_t*>(batch.columns[0][pos].data)[0]=1;
    batch.num_rows=1; batch.num_cols=2; batch.arena=&arena;
    pq::ParquetWriteOpts opts{}; opts.n_columns=2; opts.compression=0;
    for (uint32_t i=0; i<2; ++i) {
        std::strcpy(opts.columns[i].name,names[i]); opts.columns[i].type=types[i]; opts.columns[i].has_field_id=true;
    }
    opts.columns[path].field_id=2147483546; opts.columns[pos].field_id=2147483545;
    if (mode==DeleteColumns::MissingPathId) opts.columns[path].has_field_id=false;
    if (mode==DeleteColumns::MissingPositionId) opts.columns[pos].has_field_id=false;
    if (mode==DeleteColumns::DuplicatePathId) opts.columns[pos].field_id=2147483546;
    if (mode==DeleteColumns::DuplicatePositionId) opts.columns[path].field_id=2147483545;
    if (mode==DeleteColumns::InvalidPathId) opts.columns[path].field_id=-1;
    if (mode==DeleteColumns::InvalidPositionId) opts.columns[pos].field_id=7;
    auto* writer=pq::parquet_write_open_mem(&opts); if (writer==nullptr) return false;
    const bool wrote=pq::parquet_write_row_group(writer,&batch);
    const uint8_t* bytes=nullptr; uint64_t length=0;
    const bool closed=pq::parquet_write_close_mem(writer,&arena,&bytes,&length);
    if (!wrote || !closed || bytes==nullptr || length==0) return false;
    out->assign(bytes,bytes+static_cast<size_t>(length)); return true;
}

struct Fixture {
    ConditionalOrigin origin;
    Arena owner{{32u << 20,32u << 20,64,false}};
    Arena metadata{{4u << 20,4u << 20,64,false}};
    Arena source_io{{1u << 20,1u << 20,64,false}};
    Arena transport{{1u << 20,1u << 20,64,false}};
    std::array<uint8_t, 16384> workspace{};
    std::array<PinnedObject, SlotCount> pins{};
    PinnedObjectStore source{};
    ObjectStore view{};
    PinnedTableRead read{};
    TableHandle* table = nullptr;
    ScanHandle* scan = nullptr;
    std::string retained;
    uint32_t count = 0;

    ~Fixture() { if (scan != nullptr) iceberg_scan_close(scan); if (table != nullptr) iceberg_table_close(table); }
    void body(Slot slot, const std::string& value) {
        origin.objects[slot].bytes.assign(value.begin(), value.end());
    }
    bool prepare(int64_t exact_id = -1, bool absent = false, Columns mode = Columns::Valid) {
        const char* keys[SlotCount]{"retained/generation.json", "lists/exact.json", "lists/latest.json",
            "manifests/exact.json", "manifests/latest.json", "shared/exact%25part.parquet", "shared/latest.parquet",
            "manifests/deletes.json","shared/positions.parquet"};
        for (uint32_t i = 0; i < SlotCount; ++i) origin.objects[i].key = keys[i];
        if (!parquet_bytes(&origin.objects[ExactData].bytes,false,mode) ||
            !parquet_bytes(&origin.objects[LatestData].bytes,true,Columns::Valid)) return false;
        retained = metadata_json(exact_id,absent); body(MetadataSlot,retained);
        body(ExactList,manifest_list(uri("manifests/exact.json"),exact_id));
        body(LatestList,manifest_list(uri("manifests/latest.json"),19));
        body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),exact_id,origin.objects[ExactData].bytes.size()));
        body(LatestManifest,manifest(uri("shared/latest.parquet"),19,origin.objects[LatestData].bytes.size()));
        body(DeleteManifest,"[]"); body(DeleteData,"unused");
        read.selection = absent ? SnapshotSelection::Empty : SnapshotSelection::Exact;
        read.snapshot_id = absent ? 0 : exact_id;
        read.metadata_limits = {1u << 20,kRequest};
        read.parquet_metadata_bytes = 16u << 20; read.manifest_bytes = 4u << 20;
        std::strcpy(read.object.store_location,kRoot); std::strcpy(read.object.table_location,kTable);
        std::strcpy(read.object.table_uuid,kUuid); std::strcpy(read.object.object_key,keys[MetadataSlot]);
        std::strcpy(read.object.metadata_uri,uri(keys[MetadataSlot]).c_str());
        owner.set_growth_enabled(false); metadata.set_growth_enabled(false);
        source_io.set_growth_enabled(false); transport.set_growth_enabled(false);
        origin.initialize(); return true;
    }
    bool with_deletes(bool reordered=false,bool nul=false,DeleteColumns mode=DeleteColumns::Valid) {
        // Three groups make absolute position 1 belong to a later row group.
        if (!parquet_bytes(&origin.objects[ExactData].bytes,false,Columns::Valid,true)) return false;
        body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),-1,origin.objects[ExactData].bytes.size()));
        if (!delete_parquet_bytes(&origin.objects[DeleteData].bytes,reordered,nul,mode)) return false;
        std::string list=manifest_list(uri("manifests/exact.json"),-1); list.pop_back();
        std::string deletes=manifest_list(uri("manifests/deletes.json"),-1);
        const auto content=deletes.find("\"content\":0"); if (content==std::string::npos) return false;
        deletes.replace(content,11,"\"content\":1"); list+=","+deletes.substr(1); body(ExactList,list);
        std::string entry=manifest(uri("shared/positions.parquet"),-1,origin.objects[DeleteData].bytes.size());
        const auto file_content=entry.find("\"content\":0"); if (file_content==std::string::npos) return false;
        entry.replace(file_content,11,"\"content\":1");
        const auto records=entry.find("\"record_count\":3"); if (records==std::string::npos) return false;
        entry.replace(records,16,"\"record_count\":1"); body(DeleteManifest,entry); return true;
    }
    bool seal(int omit = -1) {
        count = 0;
        for (uint32_t i = 0; i < SlotCount; ++i) {
            auto& object = origin.objects[i]; object.current.size = object.bytes.size();
            std::strcpy(object.current.etag,"\"admitted-generation\"");
            if (static_cast<int>(i) == omit) continue;
            auto& pin = pins[count++]; std::strcpy(pin.key,object.key.c_str()); pin.version = object.current;
        }
        std::sort(pins.begin(),pins.begin()+count,[](const auto& a,const auto& b) { return std::strcmp(a.key,b.key)<0; });
        read.object.version = origin.objects[MetadataSlot].current;
        bolt::crypto::sha256_hex(reinterpret_cast<const uint8_t*>(retained.data()),retained.size(),read.sha256);
        PinnedObjectStoreLimits limits{SlotCount,1u << 20,kRequest};
        return pinned_object_store_init(&source,&origin.store,kRoot,pins.data(),count,&limits,&source_io,&view)==kOsOk;
    }
    bool open() {
        return iceberg_table_open_pinned(&table,&owner,&metadata,&source,&read,
            reinterpret_cast<const uint8_t*>(retained.data()),retained.size());
    }
    bool start(const ReadOptions* opts = nullptr) {
        const IcebergScanIO io{workspace.data(),workspace.size(),&transport,kRequest};
        return iceberg_scan_open_with_io(&scan,table,opts,&io);
    }
    void no_fallback() {
        EXPECT_EQ(origin.forbidden,0u); EXPECT_EQ(origin.unknown,0u);
        EXPECT_EQ(origin.objects[MetadataSlot].calls,0u); EXPECT_LE(origin.largest,kRequest);
        EXPECT_EQ(origin.objects[LatestList].calls,0u); EXPECT_EQ(origin.objects[LatestManifest].calls,0u);
        EXPECT_EQ(origin.objects[LatestData].calls,0u);
    }
};

void exact_rows(Fixture& f) {
    uint32_t rows = 0; bool finished = false;
    for (uint32_t guard = 0; guard < 8; ++guard) {
        BoltBatch batch{}; bool eof = false;
        const bool ok=iceberg_scan_next_batch(f.scan,&batch,&eof);
        if (!ok) {
            IcebergScanStats stats{};
            EXPECT_TRUE(iceberg_scan_stats(f.scan,&stats));
            std::fprintf(stderr,"pinned failure: list=%u manifest=%u data=%u visited=%llu planned=%llu\n",
                f.origin.objects[ExactList].calls,f.origin.objects[ExactManifest].calls,
                f.origin.objects[ExactData].calls,static_cast<unsigned long long>(stats.entries_visited),
                static_cast<unsigned long long>(stats.files_planned));
        }
        ASSERT_TRUE(ok);
        if (eof) { finished = true; break; }
        ASSERT_EQ(batch.num_cols,2u); ASSERT_EQ(batch.schema.num_fields,2u);
        EXPECT_STREQ(batch.schema.fields[0].name,"trade_id"); EXPECT_STREQ(batch.schema.fields[1].name,"quote");
        EXPECT_EQ(batch.schema.fields[0].type,BoltType::Int64); EXPECT_EQ(batch.schema.fields[1].type,BoltType::Float64);
        const auto* columns = batch.columns[batch.read_epoch];
        ASSERT_GE(batch.num_rows,0); ASSERT_LE(batch.num_rows,3);
        for (int64_t row = 0; row < batch.num_rows; ++row) {
            ASSERT_LT(rows,3u); EXPECT_FALSE(columns[0].is_null(row)); EXPECT_FALSE(columns[1].is_null(row));
            EXPECT_EQ(columns[0].typed_data<int64_t>()[row],41+rows);
            EXPECT_DOUBLE_EQ(columns[1].typed_data<double>()[row],(rows+1u)*1.25); ++rows;
        }
    }
    EXPECT_TRUE(finished); EXPECT_EQ(rows,3u); f.no_fallback();
}

void terminal_failure(Fixture& f) {
    BoltBatch batch{}; bool eof = true;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,&batch,&eof)); EXPECT_FALSE(eof);
    EXPECT_EQ(batch.num_rows,0); EXPECT_EQ(batch.schema.num_fields,0u);
    const uint32_t calls = f.origin.ranges;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,&batch,&eof)); EXPECT_FALSE(eof);
    EXPECT_EQ(batch.num_rows,0); EXPECT_EQ(batch.schema.num_fields,0u);
    EXPECT_EQ(f.origin.ranges,calls); f.no_fallback();
}
} // namespace

TEST(IcebergPinnedRead, RetainedHashAndCanonicalGraphReadExactSignedSnapshotByFieldId) {
    for (const int64_t id : {INT64_MIN,int64_t{-9},int64_t{-1},int64_t{0},INT64_MAX}) {
        Fixture f; ASSERT_TRUE(f.prepare(id)); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
        EXPECT_EQ(f.origin.ranges,0u);
        // Neither the retained origin body nor its current revision is consulted.
        f.origin.objects[MetadataSlot].bytes.assign(7,'x');
        std::strcpy(f.origin.objects[MetadataSlot].current.etag,"\"new-metadata\"");
        ASSERT_TRUE(f.start()); exact_rows(f);
        EXPECT_GT(f.origin.objects[ExactList].calls,1u); EXPECT_GT(f.origin.objects[ExactManifest].calls,1u);
        EXPECT_GT(f.origin.objects[ExactData].calls,1u);
    }
}

TEST(IcebergPinnedRead, MatchingExplicitOptionsPreserveMinusOneInsteadOfSelectingCurrent) {
    Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
    ReadOptions options{}; read_options_init(&options);
    options.snapshot_selection=SnapshotSelection::Exact; options.snapshot_id=-1;
    ASSERT_TRUE(f.start(&options)); exact_rows(f);
}

TEST(IcebergPinnedRead, SuppliedSelectionTimestampAndUnsupportedOptionsRefuseBeforeGraphIo) {
    for (uint32_t mode=0; mode<9; ++mode) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
        ReadOptions options{}; read_options_init(&options);
        options.snapshot_selection=SnapshotSelection::Exact; options.snapshot_id=-1;
        switch(mode) {
            case 0: options.snapshot_selection=SnapshotSelection::Legacy; break;
            case 1: options.snapshot_id=19; break;
            case 2: options.timestamp_ms=1000; break;
            case 3: options.snapshot_selection=SnapshotSelection::Empty; options.snapshot_id=0; break;
            case 4: options.n_projection=1; std::strcpy(options.projection[0],"trade_id"); break;
            case 5: options.n_predicates=1; std::strcpy(options.predicates[0].column,"trade_id"); break;
            case 6: options.include_deletes=true; break;
            case 7: options.include_partition_columns=true; break;
            default: options.snapshot_selection=static_cast<SnapshotSelection>(255); break;
        }
        EXPECT_FALSE(f.start(&options)); EXPECT_EQ(f.scan,nullptr); EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, MissingExactSnapshotCannotBecomeLatest) {
    Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); f.read.snapshot_id=-2;
    EXPECT_FALSE(f.open()); EXPECT_EQ(f.table,nullptr); EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
}

TEST(IcebergPinnedRead, ExplicitAbsenceIsEmptyAndCannotTreatNumericMinusOneAsAbsence) {
    Fixture f; ASSERT_TRUE(f.prepare(-1,true)); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); ASSERT_TRUE(f.start());
    BoltBatch batch{}; bool eof=false;
    EXPECT_TRUE(iceberg_scan_next_batch(f.scan,&batch,&eof)); EXPECT_TRUE(eof); EXPECT_EQ(f.origin.ranges,0u);
    f.no_fallback();
    Fixture numeric; ASSERT_TRUE(numeric.prepare());
    const auto at=numeric.retained.find("\"current-snapshot-id\":19"); ASSERT_NE(at,std::string::npos);
    numeric.retained.replace(at,24,"\"current-snapshot-id\":-1"); numeric.body(MetadataSlot,numeric.retained);
    ASSERT_TRUE(numeric.seal()); numeric.read.selection=SnapshotSelection::Empty; numeric.read.snapshot_id=0;
    EXPECT_FALSE(numeric.open()); EXPECT_EQ(numeric.table,nullptr); EXPECT_EQ(numeric.origin.ranges,0u); numeric.no_fallback();
}

TEST(IcebergPinnedRead, ChangedObjectsBeforeFirstTouchNeverRepinOrFallback) {
    for (const Slot slot : {ExactList,ExactManifest,ExactData}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
        std::strcpy(f.origin.objects[slot].current.etag,"\"changed-before-touch\"");
        if (slot==ExactList) { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
        else { ASSERT_TRUE(f.start()); terminal_failure(f); }
        EXPECT_EQ(f.origin.objects[slot].calls,1u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, MidReadMutationAndShortSuccessAreTerminalAtEveryGraphLevel) {
    for (const Slot slot : {ExactList,ExactManifest,ExactData}) {
        for (const bool short_success : {false,true}) {
            Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
            f.origin.fail_slot=static_cast<int>(slot); f.origin.fail_call=2; f.origin.short_success=short_success;
            if (slot==ExactList) { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
            else { ASSERT_TRUE(f.start()); terminal_failure(f); }
            EXPECT_EQ(f.origin.objects[slot].calls,2u); f.no_fallback();
        }
    }
}

TEST(IcebergPinnedRead, UnadmittedReferencesAreRejectedWithoutOriginProbe) {
    for (const Slot slot : {ExactList,ExactManifest,ExactData}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal(static_cast<int>(slot))); ASSERT_TRUE(f.open());
        if (slot==ExactList) { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
        else { ASSERT_TRUE(f.start()); terminal_failure(f); }
        EXPECT_EQ(f.origin.objects[slot].calls,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, CrossBucketOutsideRootAndRelativeReferencesCannotRebase) {
    for (const Slot level : {MetadataSlot,ExactList,ExactManifest}) {
        for (const char* invalid : {"s3://other-bucket/root space/shared/exact%part.parquet",
             "s3://read-bucket/outside/exact.parquet", "shared/exact%25part.parquet"}) {
            Fixture f; ASSERT_TRUE(f.prepare());
            if (level==MetadataSlot) {
                const auto old=uri("lists/exact.json"); const auto at=f.retained.find(old); ASSERT_NE(at,std::string::npos);
                f.retained.replace(at,old.size(),invalid); f.body(MetadataSlot,f.retained);
            } else if (level==ExactList) f.body(ExactList,manifest_list(invalid,-1));
            else f.body(ExactManifest,manifest(invalid,-1,f.origin.objects[ExactData].bytes.size()));
            ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
            if (level==MetadataSlot) { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
            else { ASSERT_TRUE(f.start()); terminal_failure(f); }
            EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
        }
    }
}

TEST(IcebergPinnedRead, RetainedDigestAndDirectoryMetadataIdentityMustAgree) {
    for (uint32_t mode=0; mode<5; ++mode) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal(mode==4 ? MetadataSlot : -1));
        if (mode==0) f.read.sha256[0]=f.read.sha256[0]=='0' ? '1' : '0';
        if (mode==1) std::strcpy(f.read.object.version.etag,"\"other-retained-pin\"");
        if (mode==2) ++f.read.object.version.size;
        if (mode==3) f.retained[0]='[';
        EXPECT_FALSE(f.open()); EXPECT_EQ(f.table,nullptr); EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, MissingDuplicateNegativeUnknownAndWrongTypeParquetIdsRefuse) {
    for (const auto mode : {Columns::MissingId,Columns::DuplicateId,Columns::NegativeId,
                           Columns::UnknownId,Columns::WrongType}) {
        Fixture f; ASSERT_TRUE(f.prepare(-1,false,mode)); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
        ASSERT_TRUE(f.start()); terminal_failure(f); EXPECT_GT(f.origin.objects[ExactData].calls,0u);
    }
}

TEST(IcebergPinnedRead, InvalidSchemaIdsRefuseBeforeOriginIo) {
    for (const char* replacement : {R"({"id":0,"name":"trade_id","required":true,"type":"long"})",
             R"({"id":11,"name":"trade_id","required":true,"type":"long"})"}) {
        Fixture f; ASSERT_TRUE(f.prepare());
        const std::string old=R"({"id":7,"name":"trade_id","required":true,"type":"long"})";
        const auto at=f.retained.find(old); ASSERT_NE(at,std::string::npos);
        f.retained.replace(at,old.size(),replacement); f.body(MetadataSlot,f.retained); ASSERT_TRUE(f.seal());
        // Either exact metadata validation or schema admission may refuse it.
        if (f.open()) { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
        EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, RequiredNestedStructBindsChildrenByIdAcrossRenameAndReorder) {
    Fixture f; ASSERT_TRUE(f.prepare()); auto& bytes=f.origin.objects[ExactData].bytes;
    ASSERT_TRUE(rewrite_footer(&bytes));
    const auto start=f.retained.find("\"fields\":[");
    const auto end=f.retained.find("],\"default-spec-id\"");
    ASSERT_NE(start,std::string::npos); ASSERT_NE(end,std::string::npos);
    f.retained.replace(start,end-start,
        R"("fields":[{"id":100,"name":"payload","required":true,"type":{"type":"struct","fields":[{"id":7,"name":"id","required":true,"type":"long"},{"id":11,"name":"price","required":true,"type":"double"}]}}]})");
    f.body(MetadataSlot,f.retained);
    f.body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),-1,bytes.size()));
    ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); ASSERT_TRUE(f.start());
    uint32_t rows=0; bool finished=false;
    for(uint32_t group=0;group<8;++group) {
        BoltBatch batch{}; bool eof=false; ASSERT_TRUE(iceberg_scan_next_batch(f.scan,&batch,&eof));
        if(eof) { finished=true;break; }
        ASSERT_EQ(batch.num_cols,1u); EXPECT_STREQ(batch.schema.fields[0].name,"payload");
        const auto& col=batch.columns[batch.read_epoch][0];
        ASSERT_EQ(col.type,BoltType::Struct); ASSERT_EQ(col.child_count(),2u);
        for(int64_t row=0;row<batch.num_rows;++row) {
            ASSERT_LT(rows,3u); EXPECT_EQ(col.child_at(0)->typed_data<int64_t>()[row],41+rows);
            EXPECT_DOUBLE_EQ(col.child_at(1)->typed_data<double>()[row],(rows+1u)*1.25); ++rows;
        }
    }
    EXPECT_TRUE(finished); EXPECT_EQ(rows,3u); f.no_fallback();
}

TEST(IcebergPinnedRead, RetainedNestedDeclarationCannotAdmitAFlatPhysicalLeaf) {
    Fixture f; ASSERT_TRUE(f.prepare());
    const std::string old=R"({"id":7,"name":"trade_id","required":true,"type":"long"})";
    const auto at=f.retained.find(old); ASSERT_NE(at,std::string::npos);
    f.retained.replace(at,old.size(),R"({"id":7,"name":"trade_id","required":true,"type":{"type":"struct","fields":[{"id":8,"name":"child","required":true,"type":"long"}]}})");
    f.body(MetadataSlot,f.retained); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
    ASSERT_TRUE(f.start()); terminal_failure(f);
}

TEST(IcebergPinnedRead, DirectoryCannotAliasTransportOrWritableReadOutputs) {
    Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
    const auto before=f.pins;
    IcebergScanIO io{f.workspace.data(),f.workspace.size(),reinterpret_cast<Arena*>(f.pins.data()),kRequest};
    EXPECT_FALSE(iceberg_scan_open_with_io(&f.scan,f.table,nullptr,&io));
    EXPECT_EQ(f.scan,nullptr); EXPECT_EQ(std::memcmp(f.pins.data(),before.data(),sizeof(before)),0);
    EXPECT_EQ(f.origin.ranges,0u);
    ASSERT_TRUE(f.start()); bool eof=true;
    EXPECT_FALSE(iceberg_scan_next_batch(f.scan,reinterpret_cast<BoltBatch*>(f.pins.data()),&eof));
    EXPECT_TRUE(eof); EXPECT_EQ(std::memcmp(f.pins.data(),before.data(),sizeof(before)),0);
    // Invalid output admission must leave the live scan usable.
    exact_rows(f);
}

TEST(IcebergPinnedRead, ExplicitFiniteBudgetsAndDistinctFrozenArenasAreRequired) {
    for (uint32_t mode=0; mode<6; ++mode) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.seal());
        if (mode==0) f.read.parquet_metadata_bytes=0;
        if (mode==1) f.read.manifest_bytes=0;
        if (mode==2) f.owner.set_growth_enabled(true);
        if (mode==3) f.metadata.set_growth_enabled(true);
        if (mode==4) f.read.selection=SnapshotSelection::Legacy;
        if (mode==5) {
            EXPECT_FALSE(iceberg_table_open_pinned(&f.table,&f.owner,&f.owner,&f.source,&f.read,
                reinterpret_cast<const uint8_t*>(f.retained.data()),f.retained.size()));
        } else EXPECT_FALSE(f.open());
        EXPECT_EQ(f.table,nullptr); EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, MalformedNumericIdentityCannotSelectAnAlternativeSnapshotOrField) {
    const std::array<std::array<const char*,2>,8> replacements{{
        {{"\"snapshot-id\":-1", "\"snapshot-id\":-1.5"}},
        {{"\"snapshot-id\":-1", "\"snapshot-id\":9223372036854775808"}},
        {{"\"snapshot-id\":-1", "\"snapshot-id\":-1,\"snapshot-id\":19"}},
        {{"\"snapshot-id\":-1,", ""}},
        {{"\"id\":7", "\"id\":7.5"}},
        {{"\"id\":7", "\"id\":2147483648"}},
        {{"\"id\":7", "\"id\":7,\"id\":11"}},
        {{"\"id\":7,", ""}}
    }};
    for (const auto& replacement : replacements) {
        Fixture f; ASSERT_TRUE(f.prepare());
        const auto at=f.retained.find(replacement[0]); ASSERT_NE(at,std::string::npos);
        f.retained.replace(at,std::strlen(replacement[0]),replacement[1]);
        f.body(MetadataSlot,f.retained); ASSERT_TRUE(f.seal());
        EXPECT_FALSE(f.open()); EXPECT_EQ(f.table,nullptr); EXPECT_EQ(f.origin.ranges,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, RequiredPhysicalGroupCannotMasqueradeAsFlatDataWithMatchingLeafIds) {
    Fixture f; ASSERT_TRUE(f.prepare()); auto& bytes=f.origin.objects[ExactData].bytes;
    ASSERT_TRUE(rewrite_footer(&bytes));
    // Prove this is a decodable nested file, rather than corrupt input that any
    // footer parser would reject. REQUIRED adds neither definition nor repeat levels.
    Arena arena; auto* meta=arena.allocate_array<pq::PqMeta>(1); ASSERT_NE(meta,nullptr);
    std::memset(meta,0,sizeof(*meta));
    ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&arena,meta));
    ASSERT_EQ(meta->n_columns,2u); EXPECT_STREQ(meta->columns[0].name,"required_group.legacy_quote");
    EXPECT_EQ(meta->columns[0].field_id,11); EXPECT_EQ(meta->columns[1].field_id,7);
    for (uint32_t c=0; c<2; ++c) { EXPECT_EQ(meta->columns[c].max_def,0); EXPECT_EQ(meta->columns[c].max_rep,0); }
    BoltColumn decoded[2]{}; int64_t rows=0;
    ASSERT_TRUE(pq::parquet_read_row_group(bytes.data(),bytes.size(),meta,0,&arena,decoded,&rows));
    ASSERT_EQ(rows,3);
    for (uint32_t i=0; i<3; ++i) {
        EXPECT_DOUBLE_EQ(decoded[0].typed_data<double>()[i],(i+1u)*1.25);
        EXPECT_EQ(decoded[1].typed_data<int64_t>()[i],41+i);
    }
    f.body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),-1,bytes.size()));
    ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); ASSERT_TRUE(f.start()); terminal_failure(f);
}

TEST(IcebergPinnedRead, RealPinnedPositionDeletesRemoveExactRowWithReservedIdsAcrossPhysicalOrderAndNames) {
    for (const bool reordered : {false,true}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes(reordered));
        {
            Arena check; auto* meta=check.allocate_array<pq::PqMeta>(1); ASSERT_NE(meta,nullptr);
            *meta=pq::PqMeta{}; const auto& bytes=f.origin.objects[DeleteData].bytes;
            ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&check,meta));
            uint64_t offset=0; uint32_t size=0;
            ASSERT_TRUE(pq::pq_locate_footer(bytes.data(),bytes.size(),&offset,&size));
            const auto status=pq::pq_parse_file_meta_strict(bytes.data()+offset,size,meta,pq::kPqMaxRowGroups);
            ASSERT_EQ(status,pq::PqMetaParseStatus::Ok);
            ASSERT_EQ(meta->n_columns,2u);
        }
        ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); ASSERT_TRUE(f.start());
        uint32_t seen=0; bool finished=false;
        const int64_t expected[2]{41,43}; const double quotes[2]{1.25,3.75};
        for (uint32_t guard=0; guard<8; ++guard) {
            BoltBatch batch{}; bool eof=false; ASSERT_TRUE(iceberg_scan_next_batch(f.scan,&batch,&eof));
            if (eof) { finished=true; break; }
            ASSERT_EQ(batch.num_cols,2u);
            EXPECT_STREQ(batch.schema.fields[0].name,"trade_id"); EXPECT_STREQ(batch.schema.fields[1].name,"quote");
            const auto* columns=batch.columns[batch.read_epoch];
            ASSERT_GE(batch.num_rows,0); ASSERT_LE(batch.num_rows,3);
            for (int64_t row=0; row<batch.num_rows; ++row) {
                ASSERT_LT(seen,2u); EXPECT_EQ(columns[0].typed_data<int64_t>()[row],expected[seen]);
                EXPECT_DOUBLE_EQ(columns[1].typed_data<double>()[row],quotes[seen]); ++seen;
            }
        }
        EXPECT_TRUE(finished); EXPECT_EQ(seen,2u); f.no_fallback();
        EXPECT_GT(f.origin.objects[DeleteManifest].calls,1u); EXPECT_GT(f.origin.objects[DeleteData].calls,1u);
        IcebergScanStats stats{}; ASSERT_TRUE(iceberg_scan_stats(f.scan,&stats)); EXPECT_EQ(stats.delete_manifests_read,1u);
    }
}

TEST(IcebergPinnedRead, MissingPositionDeletePinsFailWithoutOriginProbeOrPartialRows) {
    for (const Slot missing : {DeleteManifest,DeleteData}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes());
        ASSERT_TRUE(f.seal(static_cast<int>(missing))); ASSERT_TRUE(f.open());
        EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr);
        EXPECT_EQ(f.origin.objects[missing].calls,0u); EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, PositionDeleteManifestAndFileChangesOrShortReadsRefuseWithoutFallback) {
    for (const Slot changed : {DeleteManifest,DeleteData}) {
        for (uint32_t mode=0; mode<3; ++mode) {
            Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes()); ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
            if (mode==0) std::strcpy(f.origin.objects[changed].current.etag,"\"changed-before-delete-touch\"");
            else { f.origin.fail_slot=static_cast<int>(changed); f.origin.fail_call=2; f.origin.short_success=mode==2; }
            EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr);
            EXPECT_EQ(f.origin.objects[changed].calls,mode==0 ? 1u : 2u);
            EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
        }
    }
}

TEST(IcebergPinnedRead, EmbeddedNulInActualUtf8PositionDeletePathCannotDeletePrefixTarget) {
    for (const bool reordered : {false,true}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes(reordered,true));
        const auto& bytes=f.origin.objects[DeleteData].bytes;
        Arena arena; auto* meta=arena.allocate_array<pq::PqMeta>(1); ASSERT_NE(meta,nullptr);
        std::memset(meta,0,sizeof(*meta)); ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&arena,meta));
        BoltColumn decoded[2]{}; int64_t rows=0;
        ASSERT_TRUE(pq::parquet_read_row_group(bytes.data(),bytes.size(),meta,0,&arena,decoded,&rows)); ASSERT_EQ(rows,1);
        const uint8_t* path=nullptr; int32_t length=0; decoded[reordered ? 1 : 0].utf8_at(0,&path,&length);
        std::string expected=uri("shared/exact%25part.parquet"); expected.push_back('\0'); expected+="untrusted-tail";
        ASSERT_EQ(length,static_cast<int32_t>(expected.size()));
        EXPECT_EQ(std::memcmp(path,expected.data(),expected.size()),0);
        ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr);
        EXPECT_GT(f.origin.objects[DeleteData].calls,0u); EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, MissingDuplicateAndInvalidReservedPositionDeleteIdsCannotBindByNames) {
    for (const auto mode : {DeleteColumns::MissingPathId,DeleteColumns::MissingPositionId,
         DeleteColumns::DuplicatePathId,DeleteColumns::DuplicatePositionId,
         DeleteColumns::InvalidPathId,DeleteColumns::InvalidPositionId}) {
        Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes(false,false,mode));
        ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr);
        EXPECT_GT(f.origin.objects[DeleteData].calls,0u); EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
    }
}

TEST(IcebergPinnedRead, RequiredPhysicalGroupCannotMasqueradeAsFlatPositionDeleteColumns) {
    Fixture f; ASSERT_TRUE(f.prepare()); ASSERT_TRUE(f.with_deletes());
    auto& bytes=f.origin.objects[DeleteData].bytes; ASSERT_TRUE(rewrite_footer(&bytes));
    Arena arena; auto* meta=arena.allocate_array<pq::PqMeta>(1); ASSERT_NE(meta,nullptr);
    std::memset(meta,0,sizeof(*meta)); ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&arena,meta));
    ASSERT_EQ(meta->n_columns,2u); EXPECT_STREQ(meta->columns[0].name,"required_group.file_path");
    EXPECT_EQ(meta->columns[0].field_id,2147483546); EXPECT_EQ(meta->columns[1].field_id,2147483545);
    for (uint32_t c=0; c<2; ++c) { EXPECT_EQ(meta->columns[c].max_def,0); EXPECT_EQ(meta->columns[c].max_rep,0); }
    BoltColumn decoded[2]{}; int64_t rows=0;
    ASSERT_TRUE(pq::parquet_read_row_group(bytes.data(),bytes.size(),meta,0,&arena,decoded,&rows)); ASSERT_EQ(rows,1);
    const uint8_t* path=nullptr; int32_t length=0; decoded[0].utf8_at(0,&path,&length);
    const auto expected=uri("shared/exact%25part.parquet"); ASSERT_EQ(length,static_cast<int32_t>(expected.size()));
    EXPECT_EQ(std::memcmp(path,expected.data(),expected.size()),0); EXPECT_EQ(decoded[1].typed_data<int64_t>()[0],1);
    auto entry=manifest(uri("shared/positions.parquet"),-1,bytes.size());
    entry.replace(entry.find("\"content\":0"),11,"\"content\":1");
    entry.replace(entry.find("\"record_count\":3"),16,"\"record_count\":1"); f.body(DeleteManifest,entry);
    ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr);
    EXPECT_EQ(f.origin.objects[ExactData].calls,0u); f.no_fallback();
}

TEST(IcebergPinnedRead, MalformedLogicalAnnotationAcceptedByOrdinaryDecoderRefusesBothPinnedParquetPaths) {
    for (const Slot target : {ExactData,DeleteData}) {
        Fixture f; ASSERT_TRUE(f.prepare());
        if (target==DeleteData) ASSERT_TRUE(f.with_deletes());
        auto& bytes=f.origin.objects[target].bytes; ASSERT_TRUE(rewrite_footer(&bytes,false,true));
        Arena arena; auto* meta=arena.allocate_array<pq::PqMeta>(1); ASSERT_NE(meta,nullptr);
        std::memset(meta,0,sizeof(*meta)); ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&arena,meta));
        ASSERT_EQ(meta->n_columns,2u); EXPECT_EQ(meta->columns[1].logical,static_cast<int32_t>(pq::PqLogical::Int));
        EXPECT_EQ(meta->columns[1].int_bits,64);
        BoltColumn decoded[2]{}; int64_t rows=0;
        ASSERT_TRUE(pq::parquet_read_row_group(bytes.data(),bytes.size(),meta,0,&arena,decoded,&rows));
        EXPECT_EQ(rows,target==ExactData ? 3 : 1);
        // Physical page values remain intact despite the incomplete IntType.
        EXPECT_EQ(decoded[1].typed_data<int64_t>()[0],target==ExactData ? 41 : 1);
        if (target==ExactData) f.body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),-1,bytes.size()));
        else {
            auto entry=manifest(uri("shared/positions.parquet"),-1,bytes.size());
            entry.replace(entry.find("\"content\":0"),11,"\"content\":1");
            entry.replace(entry.find("\"record_count\":3"),16,"\"record_count\":1"); f.body(DeleteManifest,entry);
        }
        ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open());
        if (target==ExactData) { ASSERT_TRUE(f.start()); terminal_failure(f); }
        else { EXPECT_FALSE(f.start()); EXPECT_EQ(f.scan,nullptr); }
        f.no_fallback();
    }
}

TEST(IcebergPinnedRead, ExternalNestedOracle) {
    const char* parquet=std::getenv("BOLT_PINNED_NESTED_PARQUET");
    const char* schema_path=std::getenv("BOLT_PINNED_NESTED_SCHEMA");
    const char* output_path=std::getenv("BOLT_PINNED_NESTED_OUTPUT");
    if(!parquet || !schema_path || !output_path) GTEST_SKIP() << "independent fixture probe";
    Fixture f; ASSERT_TRUE(f.prepare());
    std::ifstream data(parquet,std::ios::binary), schema_file(schema_path);
    ASSERT_TRUE(data.good()); ASSERT_TRUE(schema_file.good());
    auto& bytes=f.origin.objects[ExactData].bytes;
    bytes.assign(std::istreambuf_iterator<char>(data),std::istreambuf_iterator<char>());
    const std::string schema_json((std::istreambuf_iterator<char>(schema_file)),{});
    {
        Arena check;auto* meta=check.allocate_array<pq::PqMeta>(1);ASSERT_NE(meta,nullptr);*meta=pq::PqMeta{};
        pq::PqSchemaTree tree{};ASSERT_TRUE(pq::parquet_read_meta(bytes.data(),bytes.size(),&check,meta));
        ASSERT_TRUE(pq::parquet_read_schema_tree(bytes.data(),bytes.size(),&check,&tree));
        uint64_t at=0;uint32_t size=0;ASSERT_TRUE(pq::pq_locate_footer(bytes.data(),bytes.size(),&at,&size));
        ASSERT_EQ(pq::pq_parse_file_meta_tree_strict(bytes.data()+at,size,meta,pq::kPqMaxRowGroups,&tree,&check),pq::PqMetaParseStatus::Ok);
    }

    const auto start=f.retained.find("\"schemas\":["), end=f.retained.find(",\"default-spec-id\"");
    ASSERT_NE(start,std::string::npos); ASSERT_NE(end,std::string::npos);
    f.retained.replace(start,end-start,"\"schemas\":["+schema_json+"]");
    f.body(MetadataSlot,f.retained);
    f.body(ExactManifest,manifest(uri("shared/exact%25part.parquet"),-1,bytes.size()));
    ASSERT_TRUE(f.seal()); ASSERT_TRUE(f.open()); ASSERT_TRUE(f.start());
    if(const char* call=std::getenv("BOLT_PINNED_NESTED_FAIL_CALL")) {
        f.origin.fail_slot=ExactData;f.origin.fail_call=static_cast<uint32_t>(std::strtoul(call,nullptr,10));
        f.origin.short_success=std::getenv("BOLT_PINNED_NESTED_SHORT")!=nullptr;
    }
    if(std::getenv("BOLT_PINNED_NESTED_REFUSE")) {
        terminal_failure(f);
        if(f.origin.fail_slot==ExactData) EXPECT_EQ(f.origin.objects[ExactData].calls,f.origin.fail_call);
        return;
    }
    const auto* schema=metadata_current_schema(iceberg_table_metadata(f.table)); ASSERT_NE(schema,nullptr);
    std::string json="["; uint32_t rows=0;bool finished=false;
    for(uint32_t group=0;group<128;++group) {
        BoltBatch batch{}; bool eof=false;ASSERT_TRUE(iceberg_scan_next_batch(f.scan,&batch,&eof));
        if(eof) {finished=true;break;}
        ASSERT_EQ(batch.num_cols,schema->n_fields);
        const auto* cols=batch.columns[batch.read_epoch];
        for(int64_t row=0;row<batch.num_rows;++row) {
            if(rows++)json+=",";
            json+="{";
            for(uint32_t c=0;c<schema->n_fields;++c) {
                const auto& field=schema->fields[c];if(c)json+=",";
                EXPECT_STREQ(batch.schema.fields[c].name,field.name);
                json+=nested_oracle::quote(reinterpret_cast<const uint8_t*>(field.name),std::strlen(field.name))+":";
                ASSERT_TRUE(nested_oracle::value(field,cols[c],row,0,&json)) << field.name;
            }
            json+="}";
        }
    }
    EXPECT_TRUE(finished);EXPECT_EQ(rows,3u);f.no_fallback();json+="]";
    std::ofstream output(output_path);output<<json;ASSERT_TRUE(output.good());
}
