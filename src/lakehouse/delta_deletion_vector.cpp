// bolt/lakehouse/delta_deletion_vector.cpp — Roaring-bitmap DV load + write
// (G2ICE-80).
//
// On-disk DV file format (delta-kernel's `deletion_vector.rs`, verified
// against a real `pyroaring`/CRoaring byte dump and cross-checked with the
// Rust `z85` crate's own RFC-32 test vector — see the ticket for the full
// verification trail):
//   byte 0        : version, always 1
//   [this_dv_start = dv->offset, default 1 when absent]
//   +0  4 bytes BE: dv_size  (= 4 + bitmap_len; covers magic + bitmap only)
//   +4  4 bytes LE: magic = 1681511377 (Roaring "portable" wrapper)
//   +8  dv_size-4 bytes: the RoaringBitmapArray (r64 treemap) bytes
//   +dv_size+4  4 bytes BE: CRC-32/IEEE over [magic..end of bitmap)
// `pathOrInlineDv` for storageType 'u' is the z85 (RFC 32) encoding of a raw
// 16-byte UUID (20 z85 chars, since 16 is a multiple of 4 — no padding);
// any leading characters beyond the last 20 are an optional subdirectory
// prefix. The physical file name is always
// `<prefix/>deletion_vector_<canonical-uuid-string>.bin` — the CANONICAL
// (hyphenated) UUID text, NOT the z85 string itself. Getting this wrong
// (treating the z85 string as the literal filename fragment) meant this
// reader could never have located a real DV file written by any spec-
// compliant writer (ours included, if it had used the same shortcut).

#include "bolt/lakehouse/delta/deletion_vector.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "bolt/ingest/bolt_deflate.h"    // crc32_ieee
#include "bolt/ingest/bolt_roaring.h"

namespace bolt {
namespace lakehouse {
namespace delta {

namespace {

// ROARING_BITMAP_PORTABLE_MAGIC from delta-kernel's deletion_vector.rs.
constexpr uint32_t kDvPortableMagic = 1681511377u;

// ---------------------------------------------------------------------------
// Path helpers (unchanged from before G2ICE-80).
// ---------------------------------------------------------------------------

bool join_rel(const char* prefix, const char* sub, char* out,
              uint32_t cap) noexcept {
    assert(out != nullptr && cap > 0u);
    const size_t pl = prefix != nullptr ? std::strlen(prefix) : 0u;
    const size_t sl = std::strlen(sub);
    if (pl + 1u + sl + 1u > cap) return false;
    uint32_t p = 0;
    if (pl > 0) { std::memcpy(out, prefix, pl); p = static_cast<uint32_t>(pl); }
    if (pl > 0 && prefix[pl - 1] != '/' && prefix[pl - 1] != '\\')
        out[p++] = '/';
    std::memcpy(out + p, sub, sl);
    p += static_cast<uint32_t>(sl);
    out[p] = '\0';
    return true;
}

void str_copy_cstr(char* dst, uint32_t cap, const char* src) noexcept {
    assert(dst != nullptr && cap > 0u);
    const size_t n = src != nullptr ? std::strlen(src) : 0u;
    const size_t m = n > cap - 1u ? cap - 1u : n;
    if (m > 0) std::memcpy(dst, src, m);
    dst[m] = '\0';
}

// ---------------------------------------------------------------------------
// Z85 (ZeroMQ RFC 32) — encode/decode 4-byte-aligned bytes <-> 5-char groups.
// Verified against the RFC's own "HelloWorld" test vector and against the
// Rust `z85` crate's alphabet (identical: both are the standard RFC 32
// table) before wiring this in.
// ---------------------------------------------------------------------------

constexpr char kZ85Chars[86] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".-:+=^!/*?&<>()[]{}@%$#";

bool z85_char_value(char c, uint8_t* out) noexcept {
    assert(out != nullptr);
    for (uint32_t i = 0; i < 85u; ++i) {
        if (kZ85Chars[i] == c) { *out = static_cast<uint8_t>(i); return true; }
    }
    return false;
}

bool z85_encode(const uint8_t* src, uint32_t src_len, char* dst,
                uint32_t dst_cap) noexcept {
    assert(src != nullptr && dst != nullptr);
    if ((src_len & 3u) != 0u) return false;
    const uint32_t n_chunks = src_len / 4u;
    if (n_chunks * 5u + 1u > dst_cap) return false;
    uint32_t p = 0;
    for (uint32_t c = 0; c < n_chunks; ++c) {   // bounded: n_chunks small (uuid=4)
        uint32_t v = (static_cast<uint32_t>(src[c * 4u]) << 24) |
                     (static_cast<uint32_t>(src[c * 4u + 1u]) << 16) |
                     (static_cast<uint32_t>(src[c * 4u + 2u]) << 8) |
                     static_cast<uint32_t>(src[c * 4u + 3u]);
        char chars[5];
        for (int32_t k = 4; k >= 0; --k) {
            chars[k] = kZ85Chars[v % 85u];
            v /= 85u;
        }
        for (uint32_t k = 0; k < 5u; ++k) dst[p++] = chars[k];
    }
    dst[p] = '\0';
    return true;
}

bool z85_decode(const char* src, uint32_t src_len, uint8_t* dst,
                uint32_t dst_cap) noexcept {
    assert(src != nullptr && dst != nullptr);
    if ((src_len % 5u) != 0u) return false;
    const uint32_t n_chunks = src_len / 5u;
    if (n_chunks * 4u > dst_cap) return false;
    for (uint32_t c = 0; c < n_chunks; ++c) {   // bounded: n_chunks small
        uint64_t v = 0;
        for (uint32_t k = 0; k < 5u; ++k) {
            uint8_t digit = 0;
            if (!z85_char_value(src[c * 5u + k], &digit)) return false;
            v = v * 85u + digit;
        }
        if (v > 0xFFFFFFFFull) return false;
        dst[c * 4u]      = static_cast<uint8_t>((v >> 24) & 0xFFu);
        dst[c * 4u + 1u] = static_cast<uint8_t>((v >> 16) & 0xFFu);
        dst[c * 4u + 2u] = static_cast<uint8_t>((v >> 8) & 0xFFu);
        dst[c * 4u + 3u] = static_cast<uint8_t>(v & 0xFFu);
    }
    return true;
}

// ---------------------------------------------------------------------------
// UUID generation (not cryptographically random — a time+seq+nonce mix, the
// same collision-avoidance convention `delta_writer.cpp`/`delta_delete.cpp`
// already use for part-file names) + canonical 8-4-4-4-12 string form.
// ---------------------------------------------------------------------------

std::atomic<uint64_t> g_dv_uuid_nonce{0};

uint64_t now_unix_ms() noexcept {
    return static_cast<uint64_t>(std::time(nullptr)) * 1000ull;
}

void generate_uuid_bytes(uint64_t seq, uint8_t out[16]) noexcept {
    assert(out != nullptr);
    const uint64_t nonce = g_dv_uuid_nonce.fetch_add(1, std::memory_order_relaxed);
    const uint64_t t = now_unix_ms();
    const uint64_t a = t ^ (seq * 0x9E3779B97F4A7C15ull) ^
                       (nonce * 0xC2B2AE3D27D4EB4Full);
    const uint64_t b = (t * 0xFF51AFD7ED558CCDull) ^
                       (nonce ^ 0xD6E8FEB86659FD93ull) ^ (seq << 1);
    std::memcpy(out, &a, 8);
    std::memcpy(out + 8, &b, 8);
    out[6] = static_cast<uint8_t>((out[6] & 0x0Fu) | 0x40u);  // version nibble
    out[8] = static_cast<uint8_t>((out[8] & 0x3Fu) | 0x80u);  // variant bits
}

void uuid_bytes_to_canonical(const uint8_t b[16], char out[37]) noexcept {
    assert(b != nullptr && out != nullptr);
    static const char hex[] = "0123456789abcdef";
    uint32_t p = 0;
    for (uint32_t i = 0; i < 16u; ++i) {
        if (i == 4u || i == 6u || i == 8u || i == 10u) out[p++] = '-';
        out[p++] = hex[(b[i] >> 4) & 0xFu];
        out[p++] = hex[b[i] & 0xFu];
    }
    out[p] = '\0';
    assert(p == 36u);
}

// Reconstruct the on-disk file name for storageType 'u': decode the LAST 20
// characters of `path_or_inline` (the z85-encoded UUID), format the
// canonical UUID string, and prepend any leading "prefix/" characters
// (bytes.len()-20 of them — usually zero).
bool build_uuid_file_key(const char* table_rel_prefix,
                         const char* path_or_inline, char* out,
                         uint32_t cap) noexcept {
    assert(table_rel_prefix != nullptr && path_or_inline != nullptr);
    assert(out != nullptr && cap > 0u);
    const size_t len = std::strlen(path_or_inline);
    if (len < 20u) return false;
    const size_t prefix_len = len - 20u;
    uint8_t uuid_bytes[16];
    if (!z85_decode(path_or_inline + prefix_len, 20u, uuid_bytes, sizeof(uuid_bytes)))
        return false;
    char uuid_str[37];
    uuid_bytes_to_canonical(uuid_bytes, uuid_str);
    char fname[kDeltaMaxPath];
    int n;
    if (prefix_len > 0) {
        char prefix[kDeltaMaxPath];
        if (prefix_len + 1u > sizeof(prefix)) return false;
        std::memcpy(prefix, path_or_inline, prefix_len);
        prefix[prefix_len] = '\0';
        n = std::snprintf(fname, sizeof(fname), "%s/deletion_vector_%s.bin",
                          prefix, uuid_str);
    } else {
        n = std::snprintf(fname, sizeof(fname), "deletion_vector_%s.bin",
                          uuid_str);
    }
    if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(fname)) return false;
    return join_rel(table_rel_prefix, fname, out, cap);
}

// The r64-wrapped bitmap bytes are shared by the inline and file-based
// storage types alike (both wrap a RoaringBitmapArray). Extracts bucket 0
// (the only one a `uint32_t`-row-index DV can ever populate) into `out`.
bool bitmap_from_r64_bytes(const uint8_t* bytes, uint64_t len, Arena* arena,
                           DeletionVector* out) noexcept {
    assert(arena != nullptr && out != nullptr);
    bolt::ingest::RoaringBitmap64 bm64{};
    if (!bolt::ingest::roaring_deserialize_r64(bytes, len, arena, &bm64))
        return false;
    if (bm64.n_buckets == 0) { out->present = true; return true; }  // empty DV
    if (bm64.n_buckets != 1u || bm64.buckets[0].high32 != 0u)
        return false;   // beyond this DV representation's 32-bit-row scope
    out->bitmap = bm64.buckets[0].map;
    out->present = true;
    return true;
}

uint32_t rd_u32_be(const uint8_t* p) noexcept {
    assert(p != nullptr);
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

void wr_u32_be(uint8_t* p, uint32_t v) noexcept {
    assert(p != nullptr);
    p[0] = static_cast<uint8_t>((v >> 24) & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[3] = static_cast<uint8_t>(v & 0xFFu);
}

}  // namespace

bool delta_dv_load(ObjectStore* os, const char* table_rel_prefix,
                   const DvDescriptor* dv, Arena* arena,
                   DeletionVector* out) noexcept {
    assert(os != nullptr && dv != nullptr && arena != nullptr && out != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (dv->type == DvStorageType::kNone) return true;

    if (dv->type == DvStorageType::kInline) {
        if (dv->inline_len == 0u || (dv->inline_len % 5u) != 0u) return true;
        const uint32_t raw_len = (dv->inline_len / 5u) * 4u;
        uint8_t* raw = arena->allocate_array<uint8_t>(raw_len == 0u ? 1u : raw_len);
        if (raw == nullptr) return false;
        if (!z85_decode(reinterpret_cast<const char*>(dv->inline_bytes),
                        dv->inline_len, raw, raw_len))
            return false;
        if (raw_len < 4u) return false;
        uint32_t magic = 0;
        std::memcpy(&magic, raw, 4);   // host is LE
        if (magic != kDvPortableMagic) return false;
        return bitmap_from_r64_bytes(raw + 4, raw_len - 4u, arena, out);
    }

    char key[kDeltaMaxPath];
    if (dv->type == DvStorageType::kUuid) {
        if (!build_uuid_file_key(table_rel_prefix, dv->uuid_or_path, key,
                                 sizeof(key)))
            return false;
    } else if (dv->type == DvStorageType::kPath) {
        if (dv->uuid_or_path[0] == '/') {
            str_copy_cstr(key, sizeof(key), dv->uuid_or_path);
        } else if (!join_rel(table_rel_prefix, dv->uuid_or_path, key,
                             sizeof(key))) {
            return false;
        }
    } else {
        return false;
    }

    const uint8_t* body = nullptr;
    uint64_t len = 0;
    if (os_get(os, key, arena, &body, &len) != kOsOk) return false;
    if (len < 1u || body[0] != 1u) return false;   // unknown/missing version

    const uint64_t this_dv_start = dv->offset < 0
        ? 1ull : static_cast<uint64_t>(dv->offset);
    if (this_dv_start + 8u > len) return false;
    const uint32_t dv_size = rd_u32_be(body + this_dv_start);
    if (dv->size_in_bytes >= 0 &&
        static_cast<uint64_t>(dv->size_in_bytes) != dv_size)
        return false;
    if (dv_size < 4u) return false;
    const uint64_t magic_start = this_dv_start + 4u;
    const uint64_t bitmap_start = this_dv_start + 8u;
    const uint64_t crc_start = this_dv_start + 4u + dv_size;
    if (crc_start + 4u > len) return false;

    uint32_t magic = 0;
    std::memcpy(&magic, body + magic_start, 4);   // host is LE
    if (magic != kDvPortableMagic) return false;

    const uint32_t stored_crc = rd_u32_be(body + crc_start);
    const uint32_t computed_crc =
        bolt::ingest::crc32_ieee(body + magic_start, dv_size, 0);
    if (stored_crc != computed_crc) return false;

    return bitmap_from_r64_bytes(body + bitmap_start, dv_size - 4u, arena, out);
}

bool delta_dv_contains(const DeletionVector* dv, uint64_t row_index) noexcept {
    assert(dv != nullptr);
    if (!dv->present) return false;
    if (row_index > 0xFFFFFFFFull) return false;
    return bolt::ingest::roaring_contains(
        &dv->bitmap, static_cast<uint32_t>(row_index));
}

bool delta_dv_write(ObjectStore* os, const char* table_rel_prefix,
                    const uint32_t* sorted_deleted_rows, uint64_t n_rows,
                    Arena* arena, DvDescriptor* out) noexcept {
    assert(os != nullptr && arena != nullptr && out != nullptr);
    assert(table_rel_prefix != nullptr);
    std::memset(out, 0, sizeof(*out));
    if (n_rows == 0u || sorted_deleted_rows == nullptr) return false;
    for (uint64_t i = 1; i < n_rows; ++i) {   // bounded: caller-sized n_rows
        if (sorted_deleted_rows[i] <= sorted_deleted_rows[i - 1]) return false;
    }

    const uint64_t bm_bound =
        bolt::ingest::roaring_serialize_r64_bound(n_rows);
    uint8_t* bm_bytes = arena->allocate_array<uint8_t>(bm_bound);
    if (bm_bytes == nullptr) return false;
    uint64_t bm_len = 0;
    if (!bolt::ingest::roaring_serialize_r64_single_bucket(
            sorted_deleted_rows, n_rows, arena, bm_bytes, bm_bound, &bm_len))
        return false;

    const uint32_t dv_size = 4u + static_cast<uint32_t>(bm_len);
    const uint64_t file_len = 1u + 4u + static_cast<uint64_t>(dv_size) + 4u;
    uint8_t* file = arena->allocate_array<uint8_t>(file_len);
    if (file == nullptr) return false;
    file[0] = 1u;
    wr_u32_be(file + 1, dv_size);
    const uint32_t magic = kDvPortableMagic;
    std::memcpy(file + 5, &magic, 4);   // host is LE
    std::memcpy(file + 9, bm_bytes, bm_len);
    const uint64_t crc_off = 9u + bm_len;
    const uint32_t crc = bolt::ingest::crc32_ieee(file + 5, dv_size, 0);
    wr_u32_be(file + crc_off, crc);
    assert(crc_off + 4u == file_len);

    char uuid_str[37] = {0};
    char z85_str[24] = {0};
    char rel[kDeltaMaxPath];
    char key[kDeltaMaxPath] = {0};
    bool placed = false;
    static constexpr uint32_t kMaxAttempts = 8u;
    for (uint32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
        uint8_t uuid_bytes[16];
        generate_uuid_bytes(static_cast<uint64_t>(attempt), uuid_bytes);
        uuid_bytes_to_canonical(uuid_bytes, uuid_str);
        if (!z85_encode(uuid_bytes, 16u, z85_str, sizeof(z85_str))) return false;
        const int n = std::snprintf(rel, sizeof(rel), "deletion_vector_%s.bin",
                                    uuid_str);
        if (n <= 0 || static_cast<uint32_t>(n) >= sizeof(rel)) return false;
        if (!join_rel(table_rel_prefix, rel, key, sizeof(key))) return false;
        ObjectMeta om{};
        if (os_head(os, key, &om) == kOsOk && om.exists) continue;
        placed = true;
        break;
    }
    if (!placed) return false;
    if (os_put(os, key, file, file_len) != kOsOk) return false;

    out->type = DvStorageType::kUuid;
    str_copy_cstr(out->uuid_or_path, sizeof(out->uuid_or_path), z85_str);
    out->size_in_bytes = static_cast<int64_t>(dv_size);
    out->cardinality = static_cast<int64_t>(n_rows);
    out->offset = 1;
    return true;
}

}  // namespace delta
}  // namespace lakehouse
}  // namespace bolt
