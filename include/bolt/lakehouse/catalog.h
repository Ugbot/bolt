// bolt/lakehouse/catalog.h — abstract table catalog (vtable, not virtual).
//
// A `CatalogVT` of function pointers + a `void* impl` handle. W1 ships the
// filesystem catalog: a namespace is a sub-directory of the catalog root and a
// table is a sub-directory holding a Delta `_delta_log/` (or Iceberg
// `metadata/`) tree. The vtable lists namespaces/tables, resolves a table's
// path, creates/drops tables, and commits opaque metadata bytes.
//
// Tiger Style: PODs, ≥2 asserts/fn, no exceptions, no smart pointers, bounded.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace bolt {
namespace lakehouse {

enum : int {
    kCatOk             = 0,
    kCatNotFound        = 1,
    kCatNotImplemented  = 2,
    kCatBadArg          = 3,
    kCatIoError         = 4,
    kCatExists          = 5,
};

static constexpr uint32_t kCatMaxName    = 256u;
static constexpr uint32_t kCatMaxPath     = 1024u;
static constexpr uint32_t kCatMaxListing  = 4096u;

enum class TableLayout : uint8_t { kDelta = 0, kIceberg = 1 };

struct CatalogName {
    char name[kCatMaxName];
};

struct CatalogVT {
    // List namespaces into out[0..cap); *out_n gets the count.
    int (*list_namespaces)(void* impl, CatalogName* out, uint32_t cap,
                           uint32_t* out_n) noexcept;

    // List tables within `namespace_` into out[0..cap).
    int (*list_tables)(void* impl, const char* namespace_, CatalogName* out,
                       uint32_t cap, uint32_t* out_n) noexcept;

    // Resolve the on-store path of table `namespace_.name` into out[0..cap).
    int (*table_path)(void* impl, const char* namespace_, const char* name,
                      char* out, uint32_t cap) noexcept;

    // Create table `namespace_.name` with the given layout. kCatExists if it
    // already exists. Creates the namespace dir if missing.
    int (*create_table)(void* impl, const char* namespace_, const char* name,
                        TableLayout layout) noexcept;

    // Drop table `namespace_.name` (recursive).
    int (*drop_table)(void* impl, const char* namespace_,
                      const char* name) noexcept;

    // Commit opaque metadata bytes for `namespace_.name` under `rel_path`
    // (relative to the table dir, e.g. "_delta_log/00000000000000000000.json").
    int (*commit_metadata)(void* impl, const char* namespace_,
                           const char* name, const char* rel_path,
                           const uint8_t* data, uint64_t len) noexcept;
};

struct Catalog {
    const CatalogVT* vt;
    void*            impl;
};

inline int cat_list_namespaces(Catalog* c, CatalogName* out, uint32_t cap,
                               uint32_t* n) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->list_namespaces != nullptr);
    return c->vt->list_namespaces(c->impl, out, cap, n);
}
inline int cat_list_tables(Catalog* c, const char* ns, CatalogName* out,
                           uint32_t cap, uint32_t* n) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->list_tables != nullptr);
    return c->vt->list_tables(c->impl, ns, out, cap, n);
}
inline int cat_table_path(Catalog* c, const char* ns, const char* name,
                          char* out, uint32_t cap) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->table_path != nullptr);
    return c->vt->table_path(c->impl, ns, name, out, cap);
}
inline int cat_create_table(Catalog* c, const char* ns, const char* name,
                            TableLayout layout) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->create_table != nullptr);
    return c->vt->create_table(c->impl, ns, name, layout);
}
inline int cat_drop_table(Catalog* c, const char* ns, const char* name) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->drop_table != nullptr);
    return c->vt->drop_table(c->impl, ns, name);
}
inline int cat_commit_metadata(Catalog* c, const char* ns, const char* name,
                               const char* rel, const uint8_t* d,
                               uint64_t n) noexcept {
    assert(c != nullptr && c->vt != nullptr);
    assert(c->vt->commit_metadata != nullptr);
    return c->vt->commit_metadata(c->impl, ns, name, rel, d, n);
}

// ---------------------------------------------------------------------------
// Filesystem catalog. `root` is the base directory.
// ---------------------------------------------------------------------------
static constexpr uint32_t kCatMaxRoot = 1024u;

struct FilesystemCatalog {
    char root[kCatMaxRoot];
};

// Initialise `fc` + bind `out`. Creates `root` if absent. Returns false on an
// over-long root or a directory-creation failure.
bool filesystem_catalog_init(FilesystemCatalog* fc, const char* root,
                             Catalog* out) noexcept;

}  // namespace lakehouse
}  // namespace bolt
