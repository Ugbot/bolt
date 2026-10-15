#include "bolt/interop/arrow_owner.h"
#include "bolt/bolt_arena.h"
#include <atomic>
#include <charconv>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <new>

namespace bolt::arrow {
struct Lease;
struct Context { Lease* lease; };
struct Node {
    ArrowSchema schema{};
    ArrowArray array{};
    Context schema_context{}, array_context{};
    const void* buffers[3]{};
};
struct Lease {
    alignas(64) std::atomic<bool> active{false};
    std::atomic<std::uint32_t> remaining{0};
    ArrowOwner* owner = nullptr;
    Node* nodes = nullptr;
    ArrowSchema** schemas = nullptr;
    ArrowArray** arrays = nullptr;
};
struct Stream {
    alignas(64) std::atomic<bool> active{false};
    ArrowOwner* owner = nullptr;
    bool emitted = false;
    const char* error = nullptr;
};
struct ArrowOwner {
    alignas(64) std::atomic<std::uint32_t> refs{1};
    Arena arena;
    OwnerLimits limits;
    Lease* leases = nullptr;
    Stream* streams = nullptr;
    Node* model = nullptr;
    std::uint32_t count = 0;
    bool published = false;
    ArrowOwner(const OwnerLimits& l, std::size_t capacity) noexcept
        : arena(ArenaConfig{capacity, capacity, 64, false}), limits(l) {
        arena.set_growth_enabled(false);
    }
};
namespace {
bool overlaps(const void* a, std::size_t an, const void* b, std::size_t bn) noexcept {
    assert(a != nullptr); assert(b != nullptr);
    const auto av = reinterpret_cast<std::uintptr_t>(a);
    const auto bv = reinterpret_cast<std::uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
bool output_alias(ArrowOwner* o, const void* p, std::size_t n) noexcept {
    assert(o != nullptr); assert(p != nullptr);
    return overlaps(o, sizeof(*o), p, n) || o->arena.overlaps_backing(p, n);
}
void retain(ArrowOwner* o) noexcept {
    assert(o != nullptr); assert(o->refs.load() > 0);
    o->refs.fetch_add(1, std::memory_order_relaxed);
}
void retire(Context* c) noexcept {
    assert(c != nullptr); assert(c->lease != nullptr);
    Lease* l = c->lease;
    if (l->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    ArrowOwner* o = l->owner;
    l->active.store(false, std::memory_order_release);
    owner_release(o);
}
void release_schema(ArrowSchema* s) noexcept {
    assert(s != nullptr); assert(s->private_data != nullptr);
    if (s->release == nullptr) return;
    s->release = nullptr;
    for (std::int64_t i = 0; i < s->n_children; ++i)
        if (s->children[i]->release != nullptr) s->children[i]->release(s->children[i]);
    retire(static_cast<Context*>(s->private_data));
}
void release_array(ArrowArray* a) noexcept {
    assert(a != nullptr); assert(a->private_data != nullptr);
    if (a->release == nullptr) return;
    a->release = nullptr;
    for (std::int64_t i = 0; i < a->n_children; ++i)
        if (a->children[i]->release != nullptr) a->children[i]->release(a->children[i]);
    retire(static_cast<Context*>(a->private_data));
}
Lease* acquire(ArrowOwner* o, bool schema, bool array) noexcept {
    assert(o != nullptr); assert(o->published);
    for (std::uint32_t i = 0; i < o->limits.leases; ++i) {
        Lease* l = o->leases + i;
        bool expected = false;
        if (!l->active.compare_exchange_strong(expected, true, std::memory_order_acquire)) continue;
        retain(o);
        l->remaining.store((o->count + 1) * (unsigned(schema) + unsigned(array)));
        for (std::uint32_t n = 0; n <= o->count; ++n) {
            Node& d = l->nodes[n]; const Node& m = o->model[n];
            d.schema = m.schema; d.array = m.array;
            d.schema_context.lease = l; d.array_context.lease = l;
            d.schema.private_data = &d.schema_context;
            d.array.private_data = &d.array_context;
            d.schema.release = schema ? release_schema : nullptr;
            d.array.release = array ? release_array : nullptr;
            std::memcpy(d.buffers, m.buffers, sizeof(d.buffers));
            d.array.buffers = d.buffers;
            if (n != 0) { l->schemas[n - 1] = &d.schema; l->arrays[n - 1] = &d.array; }
        }
        l->nodes[0].schema.children = l->schemas;
        l->nodes[0].array.children = l->arrays;
        return l;
    }
    return nullptr;
}
const char* copy_text(Arena* arena, const char* s, std::size_t cap) noexcept {
    assert(arena != nullptr); assert(cap > 0);
    if (s == nullptr) return nullptr;
    std::size_t len = 0;
    while (len < cap && s[len] != 0) ++len;
    if (len == cap) return nullptr;
    const auto n = len + 1;
    auto* d = arena->allocate_array<char>(n);
    if (d != nullptr) std::memcpy(d, s, n);
    return d;
}
// Validate only the flat layouts this owner promises to export. The builder
// remains responsible for payload semantics (UTF-8, null counts, alignment).
bool buffer_span(ArrowOwner* o, const void* p, std::uint64_t bytes) noexcept {
    assert(o != nullptr); assert(bytes <= o->limits.bytes);
    return bytes == 0 ? p == nullptr || o->arena.contains_backing(p, 1)
                      : p != nullptr && o->arena.contains_backing(p, static_cast<std::size_t>(bytes));
}
bool decimal128(const char* format) noexcept {
    assert(format != nullptr); assert(format[0] == 'd');
    if (format[1] != ':') return false;
    const char* end = format + std::strlen(format);
    std::int32_t precision = 0, scale = 0;
    const auto p = std::from_chars(format + 2, end, precision);
    if (p.ec != std::errc{} || p.ptr == end || *p.ptr != ',' || precision < 1 || precision > 38)
        return false;
    const auto q = std::from_chars(p.ptr + 1, end, scale);
    return q.ec == std::errc{} && q.ptr == end;
}
OwnerStatus validate_layout(ArrowOwner* o, const ArrowSchema& s,
                            const ArrowArray& a, bool root) noexcept {
    assert(o != nullptr); assert(s.format != nullptr);
    const auto rows = static_cast<std::uint64_t>(a.length);
    if (rows > o->limits.bytes) return OwnerStatus::Capacity;
    const auto bitmap = (rows + 7) / 8;
    if (root) {
        if (a.null_count != 0 || a.buffers[0] != nullptr) return OwnerStatus::Unsupported;
        return OwnerStatus::Ok;
    }
    if (std::strcmp(s.format, "n") == 0)
        return a.n_buffers == 0 && a.null_count == a.length ? OwnerStatus::Ok : OwnerStatus::Invalid;
    std::uint64_t width = 0;
    if (s.format[0] != 0 && s.format[1] == 0) {
        switch (s.format[0]) {
        case 'b': break;
        case 'c': case 'C': width = 1; break;
        case 's': case 'S': case 'e': width = 2; break;
        case 'i': case 'I': case 'f': width = 4; break;
        case 'l': case 'L': case 'g': width = 8; break;
        case 'u': case 'z': break;
        default: return OwnerStatus::Unsupported;
        }
    } else if (std::strcmp(s.format, "tdD") == 0) width = 4;
    else if (std::strcmp(s.format, "tdm") == 0) width = 8;
    else if (std::strlen(s.format) >= 4 && s.format[0] == 't' && s.format[1] == 's' &&
             std::strchr("smun", s.format[2]) != nullptr && s.format[3] == ':') width = 8;
    else if (std::strlen(s.format) == 3 && s.format[0] == 't' && s.format[1] == 'D' &&
             std::strchr("smun", s.format[2]) != nullptr) width = 8;
    else if (s.format[0] == 'd' && decimal128(s.format)) width = 16;
    else return OwnerStatus::Unsupported;
    const bool variable = s.format[0] == 'u' || s.format[0] == 'z';
    if (a.n_buffers != (variable ? 3 : 2)) return OwnerStatus::Invalid;
    if ((a.null_count > 0 && a.buffers[0] == nullptr) ||
        (a.buffers[0] != nullptr && !buffer_span(o, a.buffers[0], bitmap))) return OwnerStatus::Invalid;
    if (!variable) {
        if (width != 0 && rows > o->limits.bytes / width) return OwnerStatus::Capacity;
        return buffer_span(o, a.buffers[1], width == 0 ? bitmap : rows * width)
            ? OwnerStatus::Ok : OwnerStatus::Invalid;
    }
    if (rows >= o->limits.bytes / sizeof(std::int32_t)) return OwnerStatus::Capacity;
    if (!buffer_span(o, a.buffers[1], (rows + 1) * sizeof(std::int32_t))) return OwnerStatus::Invalid;
    // memcpy admits an otherwise valid unaligned C ABI offsets buffer safely.
    const auto* offsets = static_cast<const unsigned char*>(a.buffers[1]);
    std::int32_t previous = 0;
    std::memcpy(&previous, offsets, sizeof(previous));
    if (previous < 0) return OwnerStatus::Invalid;
    for (std::uint64_t i = 1; i <= rows; ++i) {
        std::int32_t next = 0;
        std::memcpy(&next, offsets + i * sizeof(next), sizeof(next));
        if (next < previous) return OwnerStatus::Invalid;
        previous = next;
    }
    if (static_cast<std::uint64_t>(previous) > o->limits.bytes) return OwnerStatus::Capacity;
    return buffer_span(o, a.buffers[2], static_cast<std::uint64_t>(previous))
        ? OwnerStatus::Ok : OwnerStatus::Invalid;
}
OwnerStatus copy_node(ArrowOwner* o, Node* d, const ArrowSchema& s,
                      const ArrowArray& a, bool root) noexcept {
    assert(o != nullptr); assert(d != nullptr);
    if (s.metadata != nullptr || s.dictionary != nullptr || a.dictionary != nullptr)
        return OwnerStatus::Unsupported;
    if (s.format == nullptr || a.offset != 0 || a.length < 0 || a.null_count < -1 ||
        a.null_count > a.length || a.n_buffers < 0 || a.n_buffers > 3 ||
        (a.n_buffers != 0 && a.buffers == nullptr)) return OwnerStatus::Invalid;
    if (!root && (s.n_children != 0 || a.n_children != 0)) return OwnerStatus::Unsupported;
    d->schema = s; d->array = a;
    d->schema.format = copy_text(&o->arena, s.format, 256);
    d->schema.name = copy_text(&o->arena, s.name == nullptr ? "" : s.name, 1024);
    if (d->schema.format == nullptr || d->schema.name == nullptr) return OwnerStatus::Capacity;
    const auto layout = validate_layout(o, d->schema, a, root);
    if (layout != OwnerStatus::Ok) return layout;
    d->schema.release = nullptr; d->array.release = nullptr;
    d->schema.private_data = nullptr; d->array.private_data = nullptr;
    d->schema.children = nullptr; d->array.children = nullptr;
    for (std::int64_t b = 0; b < a.n_buffers; ++b) {
        if (a.buffers[b] != nullptr && !o->arena.overlaps_backing(a.buffers[b], 1))
            return OwnerStatus::Invalid;
        d->buffers[b] = a.buffers[b];
    }
    d->array.buffers = d->buffers;
    return OwnerStatus::Ok;
}
int stream_schema(ArrowArrayStream* s, ArrowSchema* out) noexcept {
    if (s == nullptr || s->release == nullptr) return EINVAL;
    auto* p = static_cast<Stream*>(s->private_data);
    assert(p != nullptr); assert(p->active.load());
    if (out == nullptr || overlaps(s, sizeof(*s), out, sizeof(*out))) {
        p->error = "invalid Arrow schema output shell"; return EINVAL;
    }
    const auto status = owner_export_schema(p->owner, out);
    p->error = status == OwnerStatus::Ok ? nullptr : status == OwnerStatus::Capacity
        ? "Arrow descriptor lease capacity exhausted" : "invalid Arrow schema output shell";
    return status == OwnerStatus::Ok ? 0 : status == OwnerStatus::Capacity ? ENOMEM : EINVAL;
}
int stream_next(ArrowArrayStream* s, ArrowArray* out) noexcept {
    if (s == nullptr || s->release == nullptr) return EINVAL;
    auto* p = static_cast<Stream*>(s->private_data);
    assert(p != nullptr); assert(p->active.load());
    if (out == nullptr || output_alias(p->owner, out, sizeof(*out)) ||
        overlaps(s, sizeof(*s), out, sizeof(*out))) {
        p->error = "invalid Arrow array output shell"; return EINVAL;
    }
    p->error = nullptr;
    if (p->emitted) { *out = {}; return 0; }
    Lease* l = acquire(p->owner, false, true);
    if (l == nullptr) { p->error = "Arrow descriptor lease capacity exhausted"; return ENOMEM; }
    *out = l->nodes[0].array;
    p->emitted = true;
    return 0;
}
const char* stream_error(ArrowArrayStream* s) noexcept {
    if (s == nullptr || s->release == nullptr) return "released Arrow stream";
    auto* p = static_cast<Stream*>(s->private_data);
    assert(p != nullptr); assert(p->active.load());
    return p->error;
}
void stream_release(ArrowArrayStream* s) noexcept {
    if (s == nullptr || s->release == nullptr) return;
    auto* p = static_cast<Stream*>(s->private_data);
    assert(p != nullptr); assert(p->active.load());
    s->release = nullptr;
    ArrowOwner* o = p->owner;
    p->active.store(false, std::memory_order_release);
    owner_release(o);
}
} // namespace

OwnerStatus owner_create(const OwnerLimits& limits, ArrowOwner** out) noexcept {
    if (out == nullptr || limits.fields == 0 || limits.fields > kOwnerMaxFields ||
        limits.leases == 0 || limits.leases > kOwnerMaxLeases || limits.bytes > SIZE_MAX ||
        limits.bytes <= sizeof(ArrowOwner) + 64) return OwnerStatus::Invalid;
    const auto capacity = static_cast<std::size_t>(limits.bytes - sizeof(ArrowOwner)) & ~std::size_t{63};
    assert(capacity > 0); assert(capacity <= limits.bytes - sizeof(ArrowOwner));
    auto* o = new(std::nothrow) ArrowOwner(limits, capacity);
    if (o == nullptr) return OwnerStatus::Capacity;
    o->leases = o->arena.allocate_array<Lease>(limits.leases);
    o->streams = o->arena.allocate_array<Stream>(limits.leases);
    o->model = o->arena.allocate_array<Node>(limits.fields + 1);
    if (o->leases == nullptr || o->streams == nullptr || o->model == nullptr) {
        delete o; return OwnerStatus::Capacity;
    }
    bool valid = true;
    for (std::uint32_t i = 0; i < limits.leases; ++i) {
        auto* l = new(o->leases + i) Lease;
        new(o->streams + i) Stream;
        l->owner = o;
        l->nodes = o->arena.allocate_array<Node>(limits.fields + 1);
        l->schemas = o->arena.allocate_array<ArrowSchema*>(limits.fields);
        l->arrays = o->arena.allocate_array<ArrowArray*>(limits.fields);
        if (l->nodes == nullptr || l->schemas == nullptr || l->arrays == nullptr) valid = false;
        else for (std::uint32_t n = 0; n <= limits.fields; ++n) new(l->nodes + n) Node;
    }
    if (!valid) { delete o; return OwnerStatus::Capacity; }
    for (std::uint32_t n = 0; n <= limits.fields; ++n) new(o->model + n) Node;
    *out = o;
    return OwnerStatus::Ok;
}
Arena* owner_build_arena(ArrowOwner* o) noexcept {
    assert(o != nullptr); assert(!o->published);
    return o->published ? nullptr : &o->arena;
}
OwnerStatus owner_publish(ArrowOwner* o, const ArrowSchema& s, const ArrowArray& a) noexcept {
    if (o == nullptr || o->published || s.format == nullptr || std::strcmp(s.format, "+s") != 0 ||
        s.n_children <= 0 || s.n_children > o->limits.fields || a.n_children != s.n_children ||
        s.children == nullptr || a.children == nullptr || a.n_buffers != 1)
        return OwnerStatus::Invalid;
    auto status = copy_node(o, o->model, s, a, true);
    if (status != OwnerStatus::Ok) return status;
    for (std::int64_t i = 0; i < s.n_children; ++i) {
        if (s.children[i] == nullptr || a.children[i] == nullptr || a.children[i]->length != a.length)
            return OwnerStatus::Invalid;
        status = copy_node(o, o->model + i + 1, *s.children[i], *a.children[i], false);
        if (status != OwnerStatus::Ok) return status;
    }
    o->count = static_cast<std::uint32_t>(s.n_children);
    o->published = true;
    assert(o->count <= o->limits.fields); assert(o->model[0].array.length >= 0);
    return OwnerStatus::Ok;
}
OwnerStatus owner_export_schema(ArrowOwner* o, ArrowSchema* out) noexcept {
    if (o == nullptr || !o->published || out == nullptr) return OwnerStatus::Invalid;
    assert(o->count <= o->limits.fields); assert(o->refs.load() > 0);
    if (output_alias(o, out, sizeof(*out))) return OwnerStatus::Invalid;
    Lease* l = acquire(o, true, false);
    if (l == nullptr) return OwnerStatus::Capacity;
    *out = l->nodes[0].schema;
    return OwnerStatus::Ok;
}
OwnerStatus owner_export_array(ArrowOwner* o, ArrowSchema* schema, ArrowArray* array) noexcept {
    if (o == nullptr || !o->published || schema == nullptr || array == nullptr) return OwnerStatus::Invalid;
    assert(o->count <= o->limits.fields); assert(o->refs.load() > 0);
    if (output_alias(o, schema, sizeof(*schema)) || output_alias(o, array, sizeof(*array)) ||
        overlaps(schema, sizeof(*schema), array, sizeof(*array))) return OwnerStatus::Invalid;
    Lease* l = acquire(o, true, true);
    if (l == nullptr) return OwnerStatus::Capacity;
    *schema = l->nodes[0].schema; *array = l->nodes[0].array;
    return OwnerStatus::Ok;
}
OwnerStatus owner_export_stream(ArrowOwner* o, ArrowArrayStream* out) noexcept {
    if (o == nullptr || !o->published || out == nullptr) return OwnerStatus::Invalid;
    assert(o->count <= o->limits.fields); assert(o->refs.load() > 0);
    if (output_alias(o, out, sizeof(*out))) return OwnerStatus::Invalid;
    for (std::uint32_t i = 0; i < o->limits.leases; ++i) {
        Stream* p = o->streams + i; bool expected = false;
        if (!p->active.compare_exchange_strong(expected, true, std::memory_order_acquire)) continue;
        retain(o); p->owner = o; p->emitted = false; p->error = nullptr;
        *out = {stream_schema, stream_next, stream_error, stream_release, p};
        return OwnerStatus::Ok;
    }
    return OwnerStatus::Capacity;
}
void owner_release(ArrowOwner* o) noexcept {
    if (o == nullptr) return;
    assert(o->refs.load() > 0); assert(o->limits.leases > 0);
    if (o->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete o;
}
std::int64_t owner_rows(const ArrowOwner* o) noexcept {
    assert(o != nullptr); assert(o->published);
    return o->model[0].array.length;
}
std::uint32_t owner_fields(const ArrowOwner* o) noexcept {
    assert(o != nullptr); assert(o->published);
    return o->count;
}
const char* owner_field_name(const ArrowOwner* o, std::uint32_t i) noexcept {
    assert(o != nullptr); assert(o->published);
    return i < o->count ? o->model[i + 1].schema.name : nullptr;
}
} // namespace bolt::arrow
