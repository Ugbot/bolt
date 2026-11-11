// iceberg_puffin_ndv.inc — Puffin files and theta-sketch NDV (G2ICE-450).
// Included at the end of iceberg_scan.cpp (uses table_read_ref).

namespace {
inline uint32_t le32(const uint8_t* p) noexcept {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t le64(const uint8_t* p) noexcept {
    return static_cast<uint64_t>(le32(p)) | (static_cast<uint64_t>(le32(p + 4)) << 32);
}
constexpr uint8_t kPuffinMagic[4] = {'P', 'F', 'A', '1'};
}  // namespace

bool puffin_file_blobs(const uint8_t* file, uint64_t len, Arena* a, BlobMetadata** out,
                       uint32_t* n) noexcept {
    assert(a != nullptr && out != nullptr && n != nullptr);
    *out = nullptr; *n = 0;
    if (file == nullptr || len < 16u || std::memcmp(file, kPuffinMagic, 4) != 0 ||
        std::memcmp(file + len - 4u, kPuffinMagic, 4) != 0) return false;
    if ((file[len - 8u] & 1u) != 0u) return false;                  // compressed footer payload
    const uint64_t payload = le32(file + len - 12u);
    if (payload > len - 16u) return false;
    const uint64_t start = len - 12u - payload;
    if (start < 4u || std::memcmp(file + start - 4u, kPuffinMagic, 4) != 0) return false;
    return puffin_footer_parse(file + start, static_cast<uint32_t>(payload), a, out, n);
}

bool theta_sketch_estimate(const uint8_t* b, uint64_t len, double* out) noexcept {
    assert(out != nullptr);
    if (b == nullptr || len < 8u || b[2] != 3u) return false;       // family: compact theta
    const uint32_t pre = b[0] & 0x3Fu;
    const uint8_t flags = b[5];
    constexpr uint8_t kEmpty = 4u, kSingle = 32u;
    constexpr double kMaxTheta = 9223372036854775807.0;
    uint64_t count = 0, theta = 0x7FFFFFFFFFFFFFFFull;
    if (b[1] == 3u) {
        if (pre == 1u) { *out = (flags & kEmpty) != 0u ? 0.0 : (flags & kSingle) != 0u ? 1.0 : -1.0; return *out >= 0.0; }
        if (len < pre * 8u || pre < 2u || pre > 3u) return false;
        count = le32(b + 8);
        if (pre == 3u) theta = le64(b + 16);
    } else if (b[1] == 4u) {
        const uint32_t nbytes = b[4];
        if (pre < 1u || pre > 2u || nbytes < 1u || nbytes > 4u || len < pre * 8u + nbytes) return false;
        if (pre == 2u) theta = le64(b + 8);
        for (uint32_t i = 0; i < nbytes; ++i) count |= static_cast<uint64_t>(b[pre * 8u + i]) << (8u * i);
    } else {
        return false;
    }
    if (theta == 0u || theta > 0x7FFFFFFFFFFFFFFFull) return false;
    *out = static_cast<double>(count) * (kMaxTheta / static_cast<double>(theta));
    return true;
}

namespace {
// The theta blob of `sf` over exactly `field_id`, or nullptr.
const BlobMetadata* theta_blob(const BlobMetadata* blobs, uint32_t n, int32_t field_id) noexcept {
    assert(blobs != nullptr || n == 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const BlobMetadata& b = blobs[i];
        if (std::strcmp(b.type, kThetaBlobType) == 0 && b.n_fields == 1u && b.fields[0] == field_id) return &b;
    }
    return nullptr;
}
}  // namespace

bool iceberg_table_column_ndv(TableHandle* h, int64_t snapshot_id, int32_t field_id,
                              uint64_t max_file_bytes, Arena* scratch, int64_t* out) noexcept {
    assert(h != nullptr && scratch != nullptr && out != nullptr);
    *out = -1;
    const StatisticsFile* sf = metadata_statistics_for(&h->meta, snapshot_id);
    if (sf == nullptr) return false;
    const BlobMetadata* meta = theta_blob(sf->blobs, sf->n_blobs, field_id);
    if (meta == nullptr) return false;
    if (meta->ndv >= 0) { *out = meta->ndv; return true; }
    if (sf->file_size <= 0 || static_cast<uint64_t>(sf->file_size) > max_file_bytes) return false;
    const uint8_t* body = nullptr; uint64_t len = 0;
    if (!table_read_ref(h, sf->path, scratch, &body, &len) || len != static_cast<uint64_t>(sf->file_size)) return false;
    BlobMetadata* blobs = nullptr; uint32_t n = 0;
    if (!puffin_file_blobs(body, len, scratch, &blobs, &n)) return false;
    const BlobMetadata* b = theta_blob(blobs, n, field_id);
    if (b == nullptr) return false;
    if (b->ndv >= 0) { *out = b->ndv; return true; }
    if (b->codec[0] != '\0' || b->offset < 4 || b->length <= 0 ||
        static_cast<uint64_t>(b->offset) + static_cast<uint64_t>(b->length) > len) return false;
    double est = 0.0;
    if (!theta_sketch_estimate(body + b->offset, static_cast<uint64_t>(b->length), &est) || est < 0.0) return false;
    *out = static_cast<int64_t>(est + 0.5);
    return true;
}
