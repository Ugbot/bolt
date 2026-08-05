// bolt/lakehouse/delta_snapshot.cpp — replay actions into live-file set +
// per-file stats parsing + predicate/partition push-down evaluation.

#include "bolt/lakehouse/delta/snapshot.h"

#include <cstdlib>
#include <cstring>

#include "bolt/lakehouse/delta/checkpoint.h"
#include "bolt/parse/bolt_json.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace bj = bolt::parse::json;

namespace {

inline bool tok_eq(const bj::StructuralIndex* idx, int32_t cur,
                   const char* lit) noexcept {
    assert(idx != nullptr && lit != nullptr);
    if (cur < 0 || cur >= idx->token_count) return false;
    const bj::Token& t = idx->tokens[cur];
    if (t.type != bj::TokenType::Key && t.type != bj::TokenType::String)
        return false;
    const size_t n = std::strlen(lit);
    if (t.length != static_cast<int32_t>(n)) return false;
    return std::memcmp(idx->src + t.start, lit, n) == 0;
}

void copy_tok(const bj::StructuralIndex* idx, int32_t cur, char* dst,
              uint32_t cap) noexcept {
    assert(idx != nullptr && dst != nullptr);
    assert(cap > 0u);
    const bj::Token& t = idx->tokens[cur];
    const uint32_t n = t.length > static_cast<int32_t>(cap - 1u)
                           ? cap - 1u
                           : static_cast<uint32_t>(t.length);
    if (n > 0) std::memcpy(dst, idx->src + t.start, n);
    dst[n] = '\0';
}

// `bj::iter_skip_to_close` counts depth FROM the token under the cursor and
// already advances once for a scalar, so it must be called ON the Begin token.
// Advancing past Begin first leaves the cursor INSIDE the value and silently
// truncates the enclosing parse — see the long note in iceberg_metadata.cpp
// (G2FEAT-125), which is where that bug was finally caught.
bool skip_value(bj::Iterator* it) noexcept {
    assert(it != nullptr);
    assert(it->idx != nullptr);
    return bj::iter_skip_to_close(it);
}

int32_t find_file_idx(LiveFile* files, uint32_t n,
                      const char* path) noexcept {
    assert(files != nullptr && path != nullptr);
    for (uint32_t i = 0; i < n; ++i) {
        if (std::strcmp(files[i].path, path) == 0)
            return static_cast<int32_t>(i);
    }
    return -1;
}

struct ReplayCtx { Snapshot* snap; Arena* arena; };

bool apply_action(void* raw, const DeltaAction* a) noexcept {
    assert(raw != nullptr && a != nullptr);
    ReplayCtx* ctx = static_cast<ReplayCtx*>(raw);
    Snapshot* s = ctx->snap;
    switch (a->kind) {
        case ActionKind::kProtocol:
            s->protocol = a->protocol;
            s->has_protocol = true;
            break;
        case ActionKind::kMetadata:
            s->metadata = a->metadata;
            s->has_metadata = true;
            break;
        case ActionKind::kAdd: {
            const int32_t exist = find_file_idx(s->files, s->n_files,
                                                  a->add.path);
            LiveFile* slot = nullptr;
            if (exist >= 0) {
                slot = &s->files[exist];
            } else {
                if (s->n_files >= s->n_files_cap) return false;
                slot = &s->files[s->n_files++];
            }
            std::memset(slot, 0, sizeof(*slot));
            std::memcpy(slot->path, a->add.path, sizeof(slot->path));
            slot->size = a->add.size;
            slot->n_partition_values = a->add.n_partition_values;
            for (uint32_t i = 0; i < slot->n_partition_values; ++i)
                slot->partition_values[i] = a->add.partition_values[i];
            slot->has_dv = a->add.has_dv;
            slot->dv    = a->add.dv;
            delta_parse_stats(a->add.stats_json, a->add.stats_len,
                              ctx->arena, &slot->stats);
            break;
        }
        case ActionKind::kRemove: {
            const int32_t e = find_file_idx(s->files, s->n_files,
                                              a->rem.path);
            if (e >= 0) {
                const uint32_t last = s->n_files - 1u;
                if (static_cast<uint32_t>(e) != last)
                    s->files[e] = s->files[last];
                --s->n_files;
            }
            break;
        }
        default: break;
    }
    if (a->version > s->version) s->version = a->version;
    return true;
}

}  // namespace

bool delta_snapshot_build(ObjectStore* os, const char* table_rel_prefix,
                          int64_t max_version, Arena* arena,
                          Snapshot* out) noexcept {
    assert(os != nullptr && arena != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    out->version = -1;
    out->files = arena->allocate_array<LiveFile>(kLakeMaxLiveFiles);
    if (out->files == nullptr) return false;
    out->n_files_cap = kLakeMaxLiveFiles;
    ReplayCtx ctx{};
    ctx.snap = out;
    ctx.arena = arena;
    CheckpointInfo cp{};
    if (delta_checkpoint_discover(os, table_rel_prefix, max_version, arena, &cp)
        && cp.present) {
        if (!delta_checkpoint_replay(os, &cp, arena, &ctx, apply_action))
            return false;
    }
    return delta_log_walk_all(os, table_rel_prefix, max_version, arena,
                              &ctx, apply_action);
}

namespace {

bool parse_stats_object(const bj::StructuralIndex* idx, bj::Iterator* it,
                        FileStats* out) noexcept {
    assert(idx != nullptr && it != nullptr && out != nullptr);
    if (bj::iter_peek(it) != bj::TokenType::BeginObject) return false;
    bj::iter_advance(it);
    uint32_t guard = 0;
    while (bj::iter_peek(it) == bj::TokenType::Key && guard++ < 256u) {
        const int32_t key_cur = it->cursor;
        bj::iter_advance(it);
        if (tok_eq(idx, key_cur, "numRecords")) {
            int64_t v = 0;
            if (bj::iter_peek(it) == bj::TokenType::Int64) {
                bj::iter_int64(it, &v);
                bj::iter_advance(it);
            } else {
                skip_value(it);
            }
            out->num_records = v;
            out->has_num_records = true;
            continue;
        }
        const bool is_min  = tok_eq(idx, key_cur, "minValues");
        const bool is_max  = tok_eq(idx, key_cur, "maxValues");
        const bool is_null = tok_eq(idx, key_cur, "nullCount");
        if (!(is_min || is_max || is_null)) {
            skip_value(it);
            continue;
        }
        if (bj::iter_peek(it) != bj::TokenType::BeginObject) {
            skip_value(it);
            continue;
        }
        bj::iter_advance(it);
        uint32_t g2 = 0;
        while (bj::iter_peek(it) == bj::TokenType::Key && g2++ < 64u) {
            const int32_t ck = it->cursor;
            bj::iter_advance(it);
            char col[kLakeMaxColName];
            copy_tok(idx, ck, col, sizeof(col));
            int slot = -1;
            for (uint32_t i = 0; i < out->n_cols; ++i) {
                if (std::strcmp(out->col_names[i], col) == 0) {
                    slot = static_cast<int>(i); break;
                }
            }
            if (slot < 0 && out->n_cols < 16u) {
                slot = static_cast<int>(out->n_cols++);
                std::memcpy(out->col_names[slot], col, sizeof(col));
                out->null_counts[slot] = -1;
            }
            if (slot < 0) { skip_value(it); continue; }
            const bj::TokenType vt = bj::iter_peek(it);
            if (is_null) {
                if (vt == bj::TokenType::Int64) {
                    int64_t v = 0;
                    bj::iter_int64(it, &v);
                    out->null_counts[slot] = v;
                    bj::iter_advance(it);
                } else { skip_value(it); }
            } else {
                char buf[kLakeMaxValBytes];
                if (vt == bj::TokenType::String) {
                    copy_tok(idx, it->cursor, buf, sizeof(buf));
                    bj::iter_advance(it);
                } else if (vt == bj::TokenType::Int64 ||
                           vt == bj::TokenType::Float64) {
                    const bj::Token& t = idx->tokens[it->cursor];
                    const uint32_t n = t.length > static_cast<int32_t>(kLakeMaxValBytes - 1u)
                                           ? kLakeMaxValBytes - 1u
                                           : static_cast<uint32_t>(t.length);
                    std::memcpy(buf, idx->src + t.start, n);
                    buf[n] = '\0';
                    bj::iter_advance(it);
                } else {
                    skip_value(it);
                    buf[0] = '\0';
                }
                if (is_min) std::memcpy(out->min_str[slot], buf, sizeof(buf));
                else        std::memcpy(out->max_str[slot], buf, sizeof(buf));
            }
        }
        bj::iter_advance(it);
    }
    return bj::iter_peek(it) == bj::TokenType::EndObject &&
           bj::iter_advance(it);
}

}  // namespace

bool delta_parse_stats(const char* stats_json, uint32_t len, Arena* scratch,
                       FileStats* out) noexcept {
    assert(out != nullptr && scratch != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (stats_json == nullptr || len == 0 || stats_json[0] != '{') return true;
    bj::StructuralIndex idx{};
    if (!bj::build_index(reinterpret_cast<const uint8_t*>(stats_json),
                          static_cast<int32_t>(len), scratch, &idx)) {
        return true;
    }
    bj::Iterator it{};
    if (!bj::iter_init(&idx, &it)) return true;
    return parse_stats_object(&idx, &it, out);
}

namespace {

bool partition_passes(const LiveFile* f, const Predicate* p) noexcept {
    assert(f != nullptr && p != nullptr);
    if (p->op != PredicateOp::kEq) return true;
    for (uint32_t i = 0; i < f->n_partition_values; ++i) {
        if (std::strcmp(f->partition_values[i].key, p->column) != 0) continue;
        if (p->value.str_len == 0) return true;
        if (f->partition_values[i].is_null) return false;
        return std::strcmp(f->partition_values[i].value, p->value.str) == 0;
    }
    return true;
}

bool stat_passes(const LiveFile* f, const Predicate* p) noexcept {
    assert(f != nullptr && p != nullptr);
    int slot = -1;
    for (uint32_t i = 0; i < f->stats.n_cols; ++i) {
        if (std::strcmp(f->stats.col_names[i], p->column) == 0) {
            slot = static_cast<int>(i); break;
        }
    }
    if (slot < 0) return true;
    const char* lit = p->value.str_len > 0 ? p->value.str : "";
    char num_buf[kLakeMaxValBytes];
    if (p->value.str_len == 0 && p->value.type != BoltType::Utf8) {
        std::snprintf(num_buf, sizeof(num_buf), "%lld",
                      static_cast<long long>(p->value.i64));
        lit = num_buf;
    }
    const char* lo = f->stats.min_str[slot];
    const char* hi = f->stats.max_str[slot];
    const bool have_lo = lo[0] != '\0';
    const bool have_hi = hi[0] != '\0';
    switch (p->op) {
        case PredicateOp::kEq:
            if (have_lo && std::strcmp(lit, lo) < 0) return false;
            if (have_hi && std::strcmp(lit, hi) > 0) return false;
            return true;
        case PredicateOp::kLt: return !have_lo || std::strcmp(lo, lit) < 0;
        case PredicateOp::kLe: return !have_lo || std::strcmp(lo, lit) <= 0;
        case PredicateOp::kGt: return !have_hi || std::strcmp(hi, lit) > 0;
        case PredicateOp::kGe: return !have_hi || std::strcmp(hi, lit) >= 0;
        case PredicateOp::kNe:
            if (have_lo && have_hi &&
                std::strcmp(lo, lit) == 0 && std::strcmp(hi, lit) == 0) return false;
            return true;
        case PredicateOp::kIsNull:
            return f->stats.null_counts[slot] != 0;
        case PredicateOp::kIsNotNull:
            return f->stats.has_num_records
                ? f->stats.null_counts[slot] < f->stats.num_records
                : true;
    }
    return true;
}

}  // namespace

bool delta_file_passes(const LiveFile* f, const Predicate* preds,
                       uint32_t n_preds) noexcept {
    assert(f != nullptr);
    assert(preds != nullptr || n_preds == 0);
    for (uint32_t i = 0; i < n_preds; ++i) {
        if (!partition_passes(f, &preds[i])) return false;
        if (!stat_passes(f, &preds[i])) return false;
    }
    return true;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
