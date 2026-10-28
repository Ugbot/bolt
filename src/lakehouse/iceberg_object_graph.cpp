#include "bolt/lakehouse/iceberg/object_graph.h"
#include "bolt/lakehouse/iceberg/manifest.h"
#include "bolt/lakehouse/object_store/s3_uri.h"
#include "bolt/ingest/bolt_avro.h"
#include "bolt/join/bolt_swiss.h"
#include "bolt/parse/bolt_json.h"

#include <cassert>
#include <charconv>
#include <cstring>

namespace bolt::lakehouse::iceberg {
namespace {
using Status = GraphAdmissionStatus;
using Span = detail::ObjectSpan;
namespace json = bolt::parse::json;
namespace avro = bolt::ingest;
constexpr uint64_t kMaxObjectBytes = uint64_t{1} << 30;
enum class Role : uint8_t { Metadata, List, DataManifest, DeleteManifest, Data, Position, Equality };
struct Graph {
    ObjectStore* store;
    const GraphAdmissionRequest* request;
    const GraphAdmissionLimits* limits;
    GraphAdmissionWork* work;
    SwissTable index{};
    uint32_t* next = nullptr;
    Role* roles = nullptr;
    uint32_t count = 0, visits = 0;
    uint32_t heads = 0, ranged_reads = 0, reused_versions = 0, reused_bodies = 0;
    uint64_t bytes = 0;
    ManifestContent manifest_class = ManifestContent::kDataManifest;
    Status status = Status::Ok;
};

bool overlaps(Span a, Span b) noexcept {
    assert(detail::object_span_valid(a)); assert(detail::object_span_valid(b));
    return detail::object_spans_overlap(a, b);
}

bool arguments(ObjectStore* store, const GraphAdmissionRequest* request,
    const GraphAdmissionLimits* limits, GraphAdmissionWork* work,
    GraphAdmissionResult* result) noexcept {
    assert(sizeof(PinnedObject) <= SIZE_MAX / kOsPinnedMaxEntries);
    assert(sizeof(Arena) > 0);
    if (!store || !request || !limits || !work || !result) return false;
    static_assert(sizeof(PinnedObject)<=UINT64_MAX/UINT32_MAX);
    const uint64_t directory_bytes=static_cast<uint64_t>(work->objects_capacity)*sizeof(PinnedObject);
    if (request->metadata_length > SIZE_MAX || !work->objects || !store->vt || !store->impl ||
        directory_bytes > SIZE_MAX) return false;
    const Span inputs[]{{store,sizeof(*store)}, {store->vt,sizeof(*store->vt)},
        {request,sizeof(*request)}, {limits,sizeof(*limits)}, {work,sizeof(*work)},
        {request->metadata_json,static_cast<size_t>(request->metadata_length)}, {store->impl,1}};
    const Span outputs[]{{result,sizeof(*result)},
        {work->objects,static_cast<size_t>(directory_bytes)}};
    Arena* arenas[]{work->metadata,work->list,work->manifest,work->io,work->index};
    // Phase one: raw proposed Arena objects, without any Arena method calls.
    for (const Span span : inputs) if (!detail::object_span_valid(span)) return false;
    for (const Span span : outputs) if (!detail::object_span_valid(span)) return false;
    for (uint32_t i=0;i<2;++i) {
        for (const Span span : inputs) if (overlaps(outputs[i],span)) return false;
        for (uint32_t j=0;j<i;++j) if (overlaps(outputs[i],outputs[j])) return false;
    }
    for (uint32_t i=0;i<5;++i) {
        const Span object{arenas[i],sizeof(Arena)};
        if (!arenas[i] || !detail::object_span_valid(object)) return false;
        for (const Span span : inputs) if (overlaps(object,span)) return false;
        for (const Span span : outputs) if (overlaps(object,span)) return false;
        for (uint32_t j=0;j<i;++j) if (overlaps(object,{arenas[j],sizeof(Arena)})) return false;
    }
    // Phase two: every descriptor is disjoint from all public storage.
    for (Arena* arena : arenas) {
        if (arena->growth_enabled() || arena->concurrent()) return false;
        for (const Span span : inputs) if (arena->overlaps_backing(span.data,span.size)) return false;
        for (const Span span : outputs) if (arena->overlaps_backing(span.data,span.size)) return false;
        for (Arena* other : arenas) if (arena->overlaps_backing(other,sizeof(Arena))) return false;
    }
    for (uint32_t i=0;i<5;++i)
        for (uint32_t j=0;j<i;++j)
            if (arenas[i]->overlaps_backing(*arenas[j])) return false;
    return true;
}

bool limits_valid(const GraphAdmissionRequest& r, const GraphAdmissionLimits& l,
                   const GraphAdmissionWork& w) noexcept {
    assert(kOsPinnedMaxEntries == 65536); assert(kMaxObjectBytes <= SIZE_MAX);
    if (!l.max_objects || l.max_objects>kOsPinnedMaxEntries || !l.max_manifests ||
        l.max_manifests>65535 || !l.max_entries || !l.max_metadata_object_bytes ||
        l.max_metadata_object_bytes>kMaxObjectBytes || !l.max_total_metadata_bytes ||
        !l.request_bytes || l.request_bytes>kMetadataLoadMaxRequestBytes ||
        w.objects_capacity<l.max_objects || !r.metadata_json || !r.metadata_length ||
        (r.selection!=SnapshotSelection::Exact && r.selection!=SnapshotSelection::Empty)) return false;
    if (!r.metadata_limits.max_bytes || r.metadata_limits.max_bytes>=kMaxObjectBytes ||
        !r.metadata_limits.request_bytes || r.metadata_limits.request_bytes>kMetadataLoadMaxRequestBytes ||
        !os_object_version_valid(&r.object.version) || r.sha256[64]!=0) return false;
    for (uint32_t i=0;i<64;++i)
        if (!((r.sha256[i]>='0' && r.sha256[i]<='9') ||
              (r.sha256[i]>='a' && r.sha256[i]<='f'))) return false;
    return r.selection!=SnapshotSelection::Empty || r.snapshot_id==0;
}

Status transport(int rc) noexcept {
    assert(sizeof(Status)==1); assert(kOsOk==0);
    switch (rc) {
        case kOsOk: return Status::Ok;
        case kOsNotFound: return Status::NotFound;
        case kOsPreconditionFailed: return Status::Changed;
        case kOsBufferSmall: return Status::Capacity;
        case kOsBadArg: return Status::BadArg;
        case kOsNotImplemented: return Status::Unsupported;
        default: return Status::IoError;
    }
}

Status metadata_status(MetadataLoadStatus status) noexcept {
    assert(sizeof(MetadataLoadStatus)==1); assert(sizeof(Status)==1);
    switch(status) {
        case MetadataLoadStatus::Ok: return Status::Ok;
        case MetadataLoadStatus::BadArg: return Status::BadArg;
        case MetadataLoadStatus::NotFound: return Status::NotFound;
        case MetadataLoadStatus::Changed: return Status::Changed;
        case MetadataLoadStatus::IoError: return Status::IoError;
        case MetadataLoadStatus::Capacity: return Status::Capacity;
        case MetadataLoadStatus::Unsupported: return Status::Unsupported;
        case MetadataLoadStatus::IdentityMismatch: return Status::Conflict;
        default: return Status::DecodeFailed;
    }
}

Status mapped_key(const Graph& g, const char* uri, char* key) noexcept {
    assert(uri && key); assert(g.request);
    const auto rc=s3_uri_object_key(g.request->object.store_location,uri,key,kOsMaxKey);
    if (rc==S3UriStatus::Ok) return Status::Ok;
    if (rc==S3UriStatus::Capacity) return Status::Capacity;
    return rc==S3UriStatus::Unsupported ? Status::Unsupported : Status::Conflict;
}

Status lookup(Graph& g, const char* key, Role role, int64_t declared,
              bool has_extent, uint32_t* found, uint64_t* hash) noexcept {
    assert(key && found && hash); assert(g.count<=g.limits->max_objects);
    *hash=hash_bytes(key,std::strlen(key));
    const int32_t head=g.index.find(*hash);
    uint32_t link=head<0 ? UINT32_MAX : static_cast<uint32_t>(head);
    for (uint32_t step=0;step<g.count && link!=UINT32_MAX;++step) {
        assert(link<g.count);
        const auto& pin=g.work->objects[link];
        if (std::strcmp(pin.key,key)==0) {
            if (g.roles[link]!=role || (has_extent && (declared<0 ||
                static_cast<uint64_t>(declared)!=pin.version.size))) return Status::Conflict;
            *found=link; return Status::Ok;
        }
        link=g.next[link];
    }
    *found=UINT32_MAX;
    return Status::Ok;
}

Status append_pin(Graph& g, const char* key, const ObjectVersion& version,
                  Role role, uint64_t hash, uint32_t* at) noexcept {
    assert(key && at); assert(g.count<g.limits->max_objects);
    const uint32_t i=g.count;
    const int32_t previous=g.index.find(hash);
    if (!g.index.insert(hash,i)) return Status::Capacity;
    auto& pin=g.work->objects[i]; pin={};
    std::memcpy(pin.key,key,std::strlen(key)+1); pin.version=version;
    g.next[i]=previous<0 ? UINT32_MAX : static_cast<uint32_t>(previous);
    g.roles[i]=role; ++g.count; *at=i;
    assert(g.count<=g.limits->max_objects);
    return Status::Ok;
}

Status pin_reference(Graph& g, const char* uri, Role role, int64_t declared,
                     bool has_extent, uint32_t* at, bool* fresh) noexcept {
    assert(at && fresh); assert(g.work->io);
    char key[kOsMaxKey]{};
    auto rc=mapped_key(g,uri,key); if (rc!=Status::Ok) return rc;
    uint64_t hash=0;
    rc=lookup(g,key,role,declared,has_extent,at,&hash);
    if (rc!=Status::Ok) return rc;
    *fresh=*at==UINT32_MAX;
    if (!*fresh) return Status::Ok;
    if (has_extent && declared<0) return Status::Conflict;
    if (g.count==g.limits->max_objects) return Status::Capacity;
    ObjectVersion version{};
    const GraphAdmissionHints* h=g.request->hints;
    if (h && h->known_version && h->known_version(h->ctx,key,&version) && os_object_version_valid(&version) &&
        (!has_extent || version.size==static_cast<uint64_t>(declared))) {
        ++g.reused_versions;                                     // the pin the caller already holds: no HEAD
        return append_pin(g,key,version,role,hash,at);
    }
    version={}; g.work->io->reset();
    ++g.heads;
    rc=transport(os_head_versioned(g.store,key,g.work->io,&version));
    if (rc!=Status::Ok) return rc;
    if (!os_object_version_valid(&version)) return Status::Unsupported;
    if (has_extent && version.size!=static_cast<uint64_t>(declared)) return Status::Conflict;
    return append_pin(g,key,version,role,hash,at);
}

Status read_metadata(Graph& g, uint32_t at, Arena* arena, const uint8_t** out) noexcept {
    assert(at<g.count && arena && out); assert(g.bytes<=g.limits->max_total_metadata_bytes);
    const auto& pin=g.work->objects[at];
    const uint64_t size=pin.version.size;
    if (!size || size>g.limits->max_metadata_object_bytes || size>SIZE_MAX ||
        size>g.limits->max_total_metadata_bytes-g.bytes) return Status::Capacity;
    auto* bytes=arena->allocate_array<uint8_t>(static_cast<size_t>(size));
    if (!bytes) return Status::Capacity;
    g.bytes+=size;
    const GraphAdmissionHints* h=g.request->hints;
    if (h && h->known_bytes && h->known_bytes(h->ctx,pin.key,&pin.version,bytes,size)) {
        ++g.reused_bodies;                                       // exactly this version's bytes, already held: no GET
        *out=bytes; return Status::Ok;
    }
    uint64_t offset=0;
    const uint64_t steps=(size-1)/g.limits->request_bytes+1;
    for (uint64_t i=0;i<steps;++i) {
        const uint32_t n=static_cast<uint32_t>(size-offset<g.limits->request_bytes ?
            size-offset : g.limits->request_bytes);
        uint32_t got=0; g.work->io->reset();
        ++g.ranged_reads;
        const auto rc=transport(os_get_range_if_match(g.store,pin.key,&pin.version,
            offset,n,g.work->io,bytes+static_cast<size_t>(offset),&got));
        if (rc!=Status::Ok) return rc;
        if (got!=n) return Status::IoError;
        offset+=n;
    }
    assert(offset==size);
    if (h && h->on_read) h->on_read(h->ctx,pin.key,&pin.version,bytes,size);
    *out=bytes; return Status::Ok;
}

bool visit_file(void* ctx, const DataFileRef* file) noexcept {
    assert(ctx && file); auto& g=*static_cast<Graph*>(ctx);
    assert(g.visits<=g.limits->max_entries);
    if (g.visits==g.limits->max_entries) { g.status=Status::Capacity; return false; }
    ++g.visits;
    if (file->status!=ManifestStatus::kAdded && file->status!=ManifestStatus::kExisting &&
        file->status!=ManifestStatus::kDeleted) { g.status=Status::Unsupported; return false; }
    Role role=Role::Data;
    switch(file->content) {
        case FileContent::kData: role=Role::Data; break;
        case FileContent::kPositionDeletes: role=Role::Position; break;
        case FileContent::kEqualityDeletes: role=Role::Equality; break;
        default: g.status=Status::Unsupported; return false;
    }
    const bool data=g.manifest_class==ManifestContent::kDataManifest;
    if (data!=(role==Role::Data)) { g.status=Status::Conflict; return false; }
    if (file->status==ManifestStatus::kDeleted) return true;
    uint32_t at=0; bool fresh=false;
    g.status=pin_reference(g,file->file_path,role,file->stats.file_size_in_bytes,true,&at,&fresh);
    return g.status==Status::Ok;
}

// The general JSON fixture readers deliberately default absent fields and
// accept numeric coercion. Admission instead consumes critical graph fields
// through the owned SAX parser, with exact integers/presence/duplicate checks.
enum class Field : uint8_t { None, Path, Size, Content, Status, File };
struct JsonGraph {
    Graph* graph;
    ManifestListEntry* list;
    uint32_t capacity, count=0, depth=0, seen=0;
    Field selected=Field::None;
    bool list_mode=false, file_object=false, root_array=false;
    ManifestListEntry entry{};
    DataFileRef file{};
    Status status=Status::Ok;
};
constexpr uint32_t kPathBit=1, kSizeBit=2, kContentBit=4, kStatusBit=8, kFileBit=16;

JsonGraph& json_graph(void* ctx) noexcept {
    assert(ctx); auto& j=*static_cast<JsonGraph*>(ctx); assert(j.graph); return j;
}
bool json_fail(JsonGraph& j, Status status=Status::DecodeFailed) noexcept {
    assert(j.graph); assert(status!=Status::Ok); j.status=status; return false;
}
bool equal(const char* text, int32_t length, const char* expected) noexcept {
    assert(text && expected); assert(length>=0);
    return static_cast<size_t>(length)==std::strlen(expected) &&
        std::memcmp(text,expected,static_cast<size_t>(length))==0;
}
bool json_key(void* ctx, const char* key, int32_t length) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth)); j.selected=Field::None;
    const bool scope=j.depth==2 || (!j.list_mode && j.depth==3 && j.file_object);
    if (!scope) return true;
    if (length<0 || std::memchr(key,'\\',static_cast<size_t>(length))) return json_fail(j);
    uint32_t bit=0;
    if ((j.list_mode && j.depth==2) || (!j.list_mode && j.depth==3 && j.file_object)) {
        if (equal(key,length,j.list_mode ? "manifest_path" : "file_path") ||
            equal(key,length,j.list_mode ? "manifest-path" : "file-path")) {
            j.selected=Field::Path; bit=kPathBit;
        } else if (equal(key,length,j.list_mode ? "manifest_length" : "file_size_in_bytes") ||
                   equal(key,length,j.list_mode ? "manifest-length" : "file-size-in-bytes")) {
            j.selected=Field::Size; bit=kSizeBit;
        } else if (equal(key,length,"content")) { j.selected=Field::Content; bit=kContentBit; }
    } else if (j.depth==2) {
        if (equal(key,length,"status")) { j.selected=Field::Status; bit=kStatusBit; }
        else if (equal(key,length,"data_file") || equal(key,length,"data-file")) {
            j.selected=Field::File; bit=kFileBit;
        }
    }
    if (bit && (j.seen&bit)) return json_fail(j);
    j.seen|=bit; return true;
}
bool json_begin_object(void* ctx) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==0 || j.depth==1) {
        if (j.depth!=1) return json_fail(j);
        if (j.count==j.capacity) return json_fail(j,Status::Capacity);
        j.entry={}; j.file={}; j.seen=0; j.file_object=false;
    } else if (j.selected==Field::File && j.depth==2 && !j.list_mode) j.file_object=true;
    else if (j.selected!=Field::None) return json_fail(j);
    j.selected=Field::None; ++j.depth;
    return true;
}
bool json_end_object(void* ctx) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==0) return json_fail(j);
    if (j.depth==3 && j.file_object) j.file_object=false;
    if (j.depth==2) {
        const uint32_t required=j.list_mode ? kPathBit|kSizeBit : kPathBit|kSizeBit|kStatusBit|kFileBit;
        if ((j.seen&required)!=required) return json_fail(j);
        if (j.list_mode) j.list[j.count]=j.entry;
        else if (!visit_file(j.graph,&j.file)) return json_fail(j,j.graph->status);
        ++j.count;
    }
    --j.depth; j.selected=Field::None; return true;
}
bool json_begin_array(void* ctx) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==1 || j.selected!=Field::None) return json_fail(j);
    if (j.depth==0) j.root_array=true;
    j.selected=Field::None; ++j.depth; return true;
}
bool json_end_array(void* ctx) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==0) return json_fail(j);
    --j.depth; j.selected=Field::None; return true;
}
bool json_string(void* ctx, const char* value, int32_t length) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==1) return json_fail(j);
    if (j.selected==Field::None) return true;
    if (j.selected!=Field::Path || length<=0 || std::memchr(value,'\\',static_cast<size_t>(length)))
        return json_fail(j);
    char* target=j.list_mode ? j.entry.manifest_path : j.file.file_path;
    const uint32_t capacity=j.list_mode ? sizeof(j.entry.manifest_path) : sizeof(j.file.file_path);
    if (static_cast<uint32_t>(length)>=capacity || std::memchr(value,0,static_cast<size_t>(length)))
        return json_fail(j);
    std::memcpy(target,value,static_cast<size_t>(length)); target[length]=0;
    j.selected=Field::None; return true;
}
bool json_integer(void* ctx, int64_t, const char* raw, int32_t length) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    if (j.depth==1) return json_fail(j);
    if (j.selected==Field::None) return true;
    int64_t value=0;
    if (!raw || length<=0) return json_fail(j);
    const auto parsed=std::from_chars(raw,raw+length,value);
    if (parsed.ec!=std::errc{} || parsed.ptr!=raw+length) return json_fail(j);
    if (j.selected==Field::Size) {
        if (value<0) return json_fail(j,Status::Conflict);
        if (j.list_mode) j.entry.manifest_length=value; else j.file.stats.file_size_in_bytes=value;
    } else if (j.selected==Field::Status) {
        if (value<0 || value>2) return json_fail(j,Status::Unsupported);
        j.file.status=static_cast<ManifestStatus>(value);
    } else if (j.selected==Field::Content) {
        if (value<0 || value>(j.list_mode ? 1 : 2)) return json_fail(j,Status::Unsupported);
        if (j.list_mode) j.entry.content=static_cast<ManifestContent>(value);
        else j.file.content=static_cast<FileContent>(value);
    } else return json_fail(j);
    j.selected=Field::None; return true;
}
bool json_scalar(void* ctx) noexcept {
    auto& j=json_graph(ctx);
    assert(j.graph); assert(j.depth<=static_cast<uint32_t>(json::kJsonMaxDepth));
    return (j.depth!=1 && j.selected==Field::None) || json_fail(j);
}
bool json_float(void* ctx, double, const char*, int32_t) noexcept { return json_scalar(ctx); }
bool json_boolean(void* ctx, bool) noexcept { return json_scalar(ctx); }
Status parse_json(Graph& g, const uint8_t* bytes, uint64_t size,
                  ManifestListEntry* entries, uint32_t capacity) noexcept {
    assert(bytes); assert(size<=kMaxObjectBytes);
    JsonGraph j{}; j.graph=&g; j.list=entries; j.capacity=capacity; j.list_mode=entries!=nullptr;
    json::SaxHandler handler{&j,json_begin_object,json_end_object,json_begin_array,json_end_array,
        json_key,json_string,json_integer,json_float,json_boolean,json_scalar};
    if (!json::sax_parse(bytes,static_cast<int32_t>(size),&handler) || j.depth!=0 || !j.root_array)
        return j.status==Status::Ok ? Status::DecodeFailed : j.status;
    if (entries) g.visits=j.count; // transient list count; caller clears before file visits
    return Status::Ok;
}

// Validate raw Avro integers before existing manifest readers narrow content
// to uint8 enums or supply zero for absent required lengths. The IO arena is
// scratch for this bounded pass, then is reset before the normal owned reader.
struct AvroCheck {
    int32_t path=-1,size=-1,content=-1,status=-1;
    uint32_t cap=0,count=0;
    bool list=false;
    Status result=Status::Ok;
};
int32_t avro_field(const avro::AvroHeader& h, const char* name) noexcept {
    assert(name); assert(h.n_fields<=avro::kAvroMaxFields);
    int32_t result=-1;
    for (uint32_t i=0;i<h.n_fields;++i) if (std::strcmp(h.field[i].name,name)==0) {
        if (result!=-1) return -2;
        result=static_cast<int32_t>(i);
    }
    return result;
}
bool avro_int(const avro::AvroValue& value) noexcept {
    assert(sizeof(value.num.i64)==8); assert(sizeof(value.type)==1);
    return !value.is_null && (value.type==avro::AvroType::kLong || value.type==avro::AvroType::kInt);
}
bool checked_avro_row(void* ctx, const avro::AvroValue* values, uint32_t n, int64_t) noexcept {
    assert(ctx && values); auto& c=*static_cast<AvroCheck*>(ctx);
    assert(c.path>=0 && c.size>=0);
    if (c.count==c.cap) { c.result=Status::Capacity; return false; }
    ++c.count;
    if (static_cast<uint32_t>(c.path)>=n || static_cast<uint32_t>(c.size)>=n) return false;
    const auto& path=values[c.path]; const auto& size=values[c.size];
    if (path.is_null || path.type!=avro::AvroType::kString || path.bytes_len==0 ||
        path.bytes_len>=(c.list ? kIcebergMaxManifestPath : kIcebergMaxPath) ||
        !path.bytes || std::memchr(path.bytes,0,path.bytes_len) || !avro_int(size)) return false;
    if (size.num.i64<0) { c.result=Status::Conflict; return false; }
    if (c.content>=0) {
        if (static_cast<uint32_t>(c.content)>=n || !avro_int(values[c.content])) return false;
        const int64_t content=values[c.content].num.i64;
        if (content<0 || content>(c.list ? 1 : 2)) { c.result=Status::Unsupported; return false; }
    }
    if (!c.list) {
        if (c.status<0 || static_cast<uint32_t>(c.status)>=n || !avro_int(values[c.status])) return false;
        const int64_t status=values[c.status].num.i64;
        if (status<0 || status>2) { c.result=Status::Unsupported; return false; }
    }
    return true;
}
Status validate_avro(Graph& g, const uint8_t* bytes, uint64_t size,
                     bool list, uint32_t cap) noexcept {
    assert(bytes); assert(cap || !list);
    Arena* scratch=g.work->io; scratch->reset();
    auto* h=scratch->allocate_array<avro::AvroHeader>(1);
    if (!h) return Status::Capacity;
    uint64_t body=0;
    if (!avro::avro_read_header(bytes,size,scratch,h,&body)) return Status::DecodeFailed;
    AvroCheck check{}; check.list=list; check.cap=cap;
    check.path=avro_field(*h,list ? "manifest_path" : "data_file.file_path");
    check.size=avro_field(*h,list ? "manifest_length" : "data_file.file_size_in_bytes");
    check.content=avro_field(*h,list ? "content" : "data_file.content");
    check.status=list ? -1 : avro_field(*h,"status");
    if (check.path<0 || check.size<0 || check.content== -2 || (!list && check.status<0))
        return Status::DecodeFailed;
    int64_t rows=0;
    const bool ok=avro::avro_read(bytes,size,scratch,&check,checked_avro_row,&rows);
    scratch->reset();
    return ok ? Status::Ok : check.result==Status::Ok ? Status::DecodeFailed : check.result;
}
bool is_avro(const uint8_t* bytes, uint64_t size) noexcept {
    assert(bytes); assert(size>0);
    return size>=4 && std::memcmp(bytes,"Obj\1",4)==0;
}

Status parse_list(Graph& g, const uint8_t* bytes, uint64_t size,
                  ManifestListEntry* list, uint32_t* count) noexcept {
    assert(list && count); assert(g.visits==0);
    if (!is_avro(bytes,size)) {
        const auto rc=parse_json(g,bytes,size,list,g.limits->max_manifests);
        *count=g.visits; g.visits=0; return rc;
    }
    auto rc=validate_avro(g,bytes,size,true,g.limits->max_manifests);
    if (rc!=Status::Ok) return rc;
    return manifest_list_parse_avro(bytes,size,g.work->list,list,g.limits->max_manifests,count)
        ? Status::Ok : Status::DecodeFailed;
}
Status traverse_manifest(Graph& g, const ManifestListEntry& entry) noexcept {
    assert(g.count<=g.limits->max_objects); assert(g.visits<=g.limits->max_entries);
    Role role=Role::DataManifest;
    if (entry.content==ManifestContent::kDeleteManifest) role=Role::DeleteManifest;
    else if (entry.content!=ManifestContent::kDataManifest) return Status::Unsupported;
    uint32_t at=0; bool fresh=false;
    auto rc=pin_reference(g,entry.manifest_path,role,entry.manifest_length,true,&at,&fresh);
    if (rc!=Status::Ok || !fresh) return rc;
    g.work->manifest->reset();
    const uint8_t* bytes=nullptr;
    rc=read_metadata(g,at,g.work->manifest,&bytes);
    if (rc!=Status::Ok) return rc;
    g.manifest_class=entry.content;
    const uint64_t size=g.work->objects[at].version.size;
    if (!is_avro(bytes,size)) return parse_json(g,bytes,size,nullptr,g.limits->max_entries-g.visits);
    rc=validate_avro(g,bytes,size,false,g.limits->max_entries-g.visits);
    if (rc!=Status::Ok) return rc;
    uint32_t visited=0; g.status=Status::Ok;
    if (!manifest_visit_avro(bytes,size,g.work->manifest,entry.partition_spec_id,
                             visit_file,&g,&visited))
        return g.status==Status::Ok ? Status::DecodeFailed : g.status;
    return Status::Ok;
}
Status traverse(Graph& g, const Snapshot& snapshot) noexcept {
    assert(g.count==1); assert(g.visits==0);
    uint32_t at=0; bool fresh=false;
    auto rc=pin_reference(g,snapshot.manifest_list,Role::List,0,false,&at,&fresh);
    if (rc!=Status::Ok) return rc;
    assert(fresh);
    const uint8_t* bytes=nullptr;
    rc=read_metadata(g,at,g.work->list,&bytes);
    if (rc!=Status::Ok) return rc;
    auto* list=g.work->list->allocate_array<ManifestListEntry>(g.limits->max_manifests);
    if (!list) return Status::Capacity;
    uint32_t count=0;
    rc=parse_list(g,bytes,g.work->objects[at].version.size,list,&count);
    if (rc!=Status::Ok) return rc;
    for (uint32_t i=0;i<count;++i) {
        rc=traverse_manifest(g,list[i]);
        if (rc!=Status::Ok) return rc;
    }
    assert(g.count<=g.limits->max_objects); return Status::Ok;
}

void sift(PinnedObject* objects, uint32_t count, uint32_t root) noexcept {
    assert(objects); assert(count<=kOsPinnedMaxEntries);
    // At most 16 levels for the admitted 65536-entry binary heap.
    for (uint32_t level=0;level<17 && root<count/2;++level) {
        uint32_t child=root*2+1;
        if (child+1<count && std::strcmp(objects[child].key,objects[child+1].key)<0) ++child;
        if (std::strcmp(objects[root].key,objects[child].key)>=0) return;
        const PinnedObject temp=objects[root]; objects[root]=objects[child]; objects[child]=temp;
        root=child;
    }
}
void sort_directory(PinnedObject* objects, uint32_t count) noexcept {
    assert(objects && count>0); assert(count<=kOsPinnedMaxEntries);
    for (uint32_t start=count/2;start>0;--start) sift(objects,count,start-1);
    for (uint32_t remaining=count;remaining>1;--remaining) {
        const PinnedObject temp=objects[0]; objects[0]=objects[remaining-1]; objects[remaining-1]=temp;
        sift(objects,remaining-1,0);
    }
    for (uint32_t i=1;i<count;++i) assert(std::strcmp(objects[i-1].key,objects[i].key)<0);
}

Status prepare(Graph& g, MetadataLoadResult* metadata, Snapshot* snapshot) noexcept {
    assert(metadata && snapshot); assert(g.request && g.limits);
    const auto& r=*g.request;
    if (r.metadata_length>g.limits->max_metadata_object_bytes ||
        r.metadata_length>g.limits->max_total_metadata_bytes) return Status::Capacity;
    Metadata decoded{};
    auto rc=metadata_status(metadata_validate_exact(&r.object,&r.metadata_limits,
        r.metadata_json,r.metadata_length,g.work->metadata,&decoded,metadata));
    if (rc!=Status::Ok) return rc;
    if (std::memcmp(metadata->sha256,r.sha256,65)!=0) return Status::Conflict;
    if (r.selection==SnapshotSelection::Empty) {
        if (decoded.current_snapshot_present) return Status::Conflict;
    } else {
        uint32_t matches=0;
        for (uint32_t i=0;i<decoded.n_snapshots;++i) if (decoded.snapshots[i].snapshot_id==r.snapshot_id) {
            *snapshot=decoded.snapshots[i]; ++matches;
        }
        if (matches==0) return Status::NotFound;
        if (matches!=1) return Status::Conflict;
    }
    g.bytes=r.metadata_length;
    if (!SwissTable::create(&g.index,g.limits->max_objects,g.work->index)) return Status::Capacity;
    g.next=g.work->index->allocate_array<uint32_t>(g.limits->max_objects);
    g.roles=g.work->index->allocate_array<Role>(g.limits->max_objects);
    if (!g.next || !g.roles) return Status::Capacity;
    uint32_t at=0;
    return append_pin(g,r.object.object_key,r.object.version,Role::Metadata,
        hash_bytes(r.object.object_key,std::strlen(r.object.object_key)),&at);
}
} // namespace

GraphAdmissionStatus iceberg_admit_object_graph(ObjectStore* store,
    const GraphAdmissionRequest* request, const GraphAdmissionLimits* limits,
    GraphAdmissionWork* work, GraphAdmissionResult* result) noexcept {
    assert(kMetadataLoadMaxRequestBytes<=kOsVersionedRangeMaxBytes);
    assert(kMaxObjectBytes<=INT32_MAX);
    if (!arguments(store,request,limits,work,result) || !limits_valid(*request,*limits,*work))
        return Status::BadArg;
    if (request->selection!=SnapshotSelection::Empty &&
        (!store->vt->head_versioned || !store->vt->get_range_if_match)) return Status::Unsupported;
    Arena* arenas[]{work->metadata,work->list,work->manifest,work->io,work->index};
    for (Arena* arena : arenas) arena->reset();
    Graph g{}; g.store=store; g.request=request; g.limits=limits; g.work=work;
    MetadataLoadResult metadata{}; Snapshot snapshot{};
    auto rc=prepare(g,&metadata,&snapshot);
    if (rc!=Status::Ok) return rc;
    if (request->selection==SnapshotSelection::Exact) {
        rc=traverse(g,snapshot); if (rc!=Status::Ok) return rc;
    }
    sort_directory(work->objects,g.count);
    GraphAdmissionResult staged{}; staged.metadata=metadata;
    staged.selection=request->selection; staged.snapshot_id=request->snapshot_id;
    staged.objects=work->objects; staged.object_count=g.count; staged.total_metadata_bytes=g.bytes;
    staged.heads=g.heads; staged.ranged_reads=g.ranged_reads;
    staged.reused_versions=g.reused_versions; staged.reused_bodies=g.reused_bodies;
    *result=staged;
    assert(result->objects==work->objects); assert(result->object_count>0);
    return Status::Ok;
}
} // namespace bolt::lakehouse::iceberg
