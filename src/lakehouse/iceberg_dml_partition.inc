// bolt/lakehouse/iceberg_dml_partition.inc — position deletes of a partitioned table (G2ICE-292). Included inside
// iceberg_writer.cpp at namespace scope after iceberg_dml.inc; not a standalone TU.
//
// A position-delete file is scoped to a partition (spec: "Scan Planning": a position delete applies to data files of
// the same partition and spec, unless its spec is unpartitioned), so the (file_path, pos) rows of one change are
// written as one delete file per (spec, tuple) of the data files they name, each stamped with that spec id and tuple.

namespace {

// The live data files of the current snapshot, findable by path.
struct LiveData {
    DataFileRef* refs;
    uint32_t     n, cap;
    const char** paths;
    RmIndex      ix;
};

bool live_data_build(TableHandle* th, LiveData* out) noexcept {
    assert(th != nullptr && out != nullptr);
    *out = LiveData{};
    ManifestListEntry* ml = nullptr; uint32_t nm = 0;
    if (!parent_list(th, th->meta.current_snapshot_id, &ml, &nm)) return false;
    for (uint32_t m = 0; m < nm; ++m) {                                   // bounded by nm
        if (ml[m].content != ManifestContent::kDataManifest) continue;
        DataFileRef* e = nullptr; uint32_t ne = 0;
        if (!read_manifest_entries(th, ml[m], &e, &ne)) return false;
        for (uint32_t i = 0; i < ne; ++i) {                               // bounded by ne
            if (e[i].status == ManifestStatus::kDeleted || e[i].content != FileContent::kData) continue;
            if (!lake_grow(th->arena, nullptr, &out->refs, out->n, &out->cap, out->n + 1u)) return false;
            out->refs[out->n] = e[i];
            if (out->refs[out->n].partition_spec_id == 0 && ml[m].partition_spec_id != 0)
                out->refs[out->n].partition_spec_id = ml[m].partition_spec_id;
            ++out->n;
        }
    }
    out->paths = th->arena->allocate_array<const char*>(out->n + 1u);
    if (out->paths == nullptr) return false;
    for (uint32_t i = 0; i < out->n; ++i) out->paths[i] = out->refs[i].file_path;   // bounded by n
    return rm_index_build(th->arena, out->paths, out->n, &out->ix);
}

const DataFileRef* live_data_find(const LiveData* l, const char* path) noexcept {
    assert(l != nullptr && path != nullptr);
    const int64_t k = rm_index_find(&l->ix, path);
    return k < 0 ? nullptr : &l->refs[k];
}

// The delete rows of one (spec, tuple).
struct DelGroup {
    const DataFileRef* like;      // a data file of the group (its spec and tuple); null = unknown file, unpartitioned
    const char**       paths;
    int64_t*           pos;
    uint32_t           n, cap_p, cap_q;
    uint32_t           _pad;
};

int64_t del_group_of(TableHandle* th, DelGroup** gs, uint32_t* ng, uint32_t* cap, SwissTableGrowable* ix,
                     const DataFileRef* like) noexcept {
    assert(th != nullptr && gs != nullptr && ng != nullptr && ix != nullptr);
    const uint32_t np = like == nullptr ? 0u : like->n_partition;
    const int32_t spec = like == nullptr ? -1 : like->partition_spec_id;
    uint64_t key = like == nullptr ? 0u : tuple_hash(like->partition, np, spec);
    for (uint32_t probe = 0; probe <= *ng; ++probe, ++key) {              // bounded: one probe per existing group
        const int32_t g = ix->find(key);
        if (g < 0) break;
        const DataFileRef* o = (*gs)[g].like;
        if ((o == nullptr) != (like == nullptr)) continue;
        if (o == nullptr || (o->partition_spec_id == spec && o->n_partition == np && tuple_equal(o->partition, like->partition, np)))
            return g;
    }
    if (!lake_grow(th->arena, nullptr, gs, *ng, cap, *ng + 1u) || !ix->insert(key, *ng)) return -1;
    (*gs)[*ng] = DelGroup{};
    (*gs)[*ng].like = like;
    return static_cast<int64_t>((*ng)++);
}

// (paths[i], pos[i]) in file_path then pos order, into arena arrays.
bool sort_position_deletes(Arena* a, const char* const* paths, const int64_t* pos, uint32_t n, const char*** out_paths,
                           int64_t** out_pos) noexcept {
    assert(a != nullptr && paths != nullptr && pos != nullptr);
    assert(out_paths != nullptr && out_pos != nullptr);
    auto* order = a->allocate_array<uint32_t>(n + 1u);
    *out_paths = a->allocate_array<const char*>(n + 1u);
    *out_pos = a->allocate_array<int64_t>(n + 1u);
    if (order == nullptr || *out_paths == nullptr || *out_pos == nullptr) return false;
    for (uint32_t i = 0; i < n; ++i) order[i] = i;                        // bounded by n
    for (uint32_t gap = n / 2u; gap > 0u; gap /= 2u)                      // shell sort: bounded, no recursion
        for (uint32_t i = gap; i < n; ++i) {
            const uint32_t x = order[i];
            uint32_t j = i;
            while (j >= gap) {
                const int c = std::strcmp(paths[order[j - gap]], paths[x]);
                if (c < 0 || (c == 0 && pos[order[j - gap]] <= pos[x])) break;
                order[j] = order[j - gap];
                j -= gap;
            }
            order[j] = x;
        }
    for (uint32_t i = 0; i < n; ++i) { (*out_paths)[i] = paths[order[i]]; (*out_pos)[i] = pos[order[i]]; }   // bounded
    return true;
}

// One delete file per (spec, tuple) of the rows' data files. Rows stay in their given (file_path, pos) order.
bool write_grouped_deletes(TableHandle* th, int64_t snap_id, const char* const* paths, const int64_t* pos, uint32_t n,
                           const LiveData* live, DataFileRef** out, char (**rels)[128], uint32_t* n_out) noexcept {
    assert(th != nullptr && live != nullptr && out != nullptr && rels != nullptr && n_out != nullptr);
    assert(paths != nullptr || n == 0u);
    *n_out = 0;
    const PartitionSpec* cur = metadata_spec(&th->meta, th->meta.current_spec_id);
    DelGroup* gs = nullptr; uint32_t ng = 0, cap = 0;
    SwissTableGrowable ix{};
    if (!SwissTableGrowable::create(&ix, 8u, th->arena)) return false;
    int64_t g = -1;
    for (uint32_t i = 0; i < n; ++i) {                                    // bounded by n
        if (i == 0u || std::strcmp(paths[i], paths[i - 1u]) != 0) {
            const DataFileRef* like = live_data_find(live, paths[i]);
            if (like == nullptr && cur != nullptr && cur->n_fields != 0u) {
                std::fprintf(stderr, "[iceberg_writer] position delete of %s, which is not a live data file of this partitioned table; nothing was written\n", paths[i]);
                return false;
            }
            g = del_group_of(th, &gs, &ng, &cap, &ix, like);
            if (g < 0) return false;
        }
        DelGroup& d = gs[g];
        if (!lake_grow(th->arena, nullptr, &d.paths, d.n, &d.cap_p, d.n + 1u) ||
            !lake_grow(th->arena, nullptr, &d.pos, d.n, &d.cap_q, d.n + 1u)) return false;
        d.paths[d.n] = paths[i];
        d.pos[d.n++] = pos[i];
    }
    if (ng > 1u && th->file_io != nullptr)
        return refuse_named(th, "position deletes spanning more than one partition through FileWriteIO (one staged file per commit)");
    *out = th->arena->allocate_array<DataFileRef>(ng + 1u);
    *rels = th->arena->allocate_array<char[128]>(ng + 1u);
    if (*out == nullptr || *rels == nullptr) return false;
    for (uint32_t k = 0; k < ng; ++k) {                                   // bounded by ng
        BoltBatch b{};
        if (!pos_delete_batch(th->arena, gs[k].paths, gs[k].pos, gs[k].n, &b)) return false;
        DataFileRef& df = (*out)[*n_out];
        if (!write_pos_delete_batch(th, snap_id, &b, (*rels)[k], &df)) return false;
        ++*n_out;
        if (gs[k].like != nullptr) {                                      // the data file's spec and tuple
            df.partition_spec_id = gs[k].like->partition_spec_id;
            df.n_partition = gs[k].like->n_partition;
            std::memcpy(df.partition, gs[k].like->partition, sizeof(PartitionValue) * df.n_partition);
        }
    }
    return true;
}

}  // namespace
