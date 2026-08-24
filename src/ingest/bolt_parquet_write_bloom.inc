// bolt_parquet_write_bloom.inc — split-block bloom filter (SBBF) builder.
//
// Included ONCE from bolt_parquet_write.cpp inside its anonymous namespace,
// after the thrift codec and the RowVal helpers. A size seam, not a public
// header.
//
// This is the exact inverse of bolt_parquet_bloom.h's reader: same 256-bit
// blocks, same fastrange block selection, same eight salts, same XXH64 over
// the PLAIN-encoded value bytes. The two are deliberately written against
// one another -- a filter is a claim that a value is ABSENT, so a builder
// that disagrees with the probe by one bit turns a pruning optimisation into
// dropped rows. test_bolt_parquet_write_bloom asserts the round trip both
// ways: every value inserted must probe present, and a large disjoint set
// must probe absent at close to the configured false-positive rate.
//
// Placement: a row group's filters are flushed immediately after that row
// group's data, which is parquet-mr's default (BloomFilterPosition
// AFTER_ROWGROUP). That bounds live filter memory to one row group's worth
// rather than the whole file's, which matters at 256 columns.

// parquet-format BloomFilter.md block_check salts. Identical to the
// reader's kSalt; duplicated rather than shared because the reader's copy is
// function-local static in another translation unit, and a bloom filter is
// exactly the place not to introduce a silent divergence -- so the test
// pins them against each other instead.
constexpr std::uint32_t kBloomSalt[8] = {
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
    0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
};

// Per-filter ceiling. parquet-mr's default NDV of 1M at fpp 0.05 sizes to
// ~860 KiB, so 1 MiB is the effective default there too; the cap keeps a
// wildly wrong NDV estimate from allocating unboundedly.
constexpr std::uint32_t kPwBloomMinBytes = 32u;

struct BloomBuilder {
    std::vector<std::uint8_t> bits;
    std::uint32_t             n_bytes;
    bool                      active;
};

// parquet-mr BlockSplitBloomFilter.optimalNumOfBits, rounded up to whole
// 256-bit blocks and clamped. The formula is
//   m = -8n / ln(1 - fpp^(1/8))
// -- eight bits are set per value, all within one block, so the per-bit
// probability is fpp^(1/8) rather than the textbook single-hash form.
std::uint32_t bloom_optimal_bytes(std::uint64_t ndv, double fpp,
                                  std::uint32_t cap) noexcept {
    assert(cap >= kPwBloomMinBytes);
    if (!(fpp > 0.0) || !(fpp < 1.0)) fpp = 0.05;
    if (ndv == 0u) return kPwBloomMinBytes;
    const double m = -8.0 * static_cast<double>(ndv) /
                     std::log(1.0 - std::pow(fpp, 1.0 / 8.0));
    double bytes = m / 8.0;
    if (!(bytes > 0.0)) bytes = static_cast<double>(cap);
    if (bytes > static_cast<double>(cap)) bytes = static_cast<double>(cap);
    std::uint32_t nb = static_cast<std::uint32_t>(bytes);
    nb = (nb + 31u) & ~31u;                     // whole 256-bit blocks
    if (nb < kPwBloomMinBytes) nb = kPwBloomMinBytes;
    if (nb > cap) nb = cap & ~31u;
    if (nb < kPwBloomMinBytes) nb = kPwBloomMinBytes;
    assert(nb >= kPwBloomMinBytes && (nb % 32u) == 0u);
    return nb;
}

void bloom_reset(BloomBuilder* b, std::uint32_t n_bytes) noexcept {
    assert(b != nullptr);
    assert(n_bytes >= kPwBloomMinBytes && (n_bytes % 32u) == 0u);
    b->bits.assign(n_bytes, 0u);
    b->n_bytes = n_bytes;
    b->active = true;
}

// Set the eight bits this key claims. The inverse of pq_bloom_may_contain:
// block chosen by fastrange over the TOP 32 bits, then one bit per word
// selected by (low32 * salt[w]) >> 27. Branch-free -- eight unconditional
// read-modify-writes into one 32-byte block, which is one cache line pair.
void bloom_insert(BloomBuilder* b, std::uint64_t key) noexcept {
    assert(b != nullptr);
    assert(b->n_bytes >= 32u && (b->n_bytes % 32u) == 0u);
    const std::uint32_t n_blocks = b->n_bytes / 32u;
    const std::uint64_t block_idx =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key >> 32)) *
         static_cast<std::uint64_t>(n_blocks)) >> 32;
    assert(block_idx < n_blocks);
    std::uint8_t* block = b->bits.data() + block_idx * 32u;
    const std::uint32_t x = static_cast<std::uint32_t>(key);
    for (std::uint32_t w = 0; w < 8u; ++w) {
        std::uint32_t word;
        std::memcpy(&word, block + 4u * w, 4);
        word |= (1u << ((x * kBloomSalt[w]) >> 27));
        std::memcpy(block + 4u * w, &word, 4);
    }
}

// BOOLEAN is excluded: parquet-mr does not write bloom filters for it and a
// two-valued domain makes one pointless.
bool bloom_eligible(BoltType t) noexcept {
    switch (t) {
        case BoltType::Int32:
        case BoltType::Date32:
        case BoltType::Float32:
        case BoltType::Int64:
        case BoltType::Timestamp:
        case BoltType::Float64:
        case BoltType::Decimal128:
        case BoltType::Utf8:
        case BoltType::Binary:
            return true;
        default:
            return false;
    }
}

// Hash one row's value the way the spec defines: XXH64 over the value's
// PLAIN encoding, seed 0. For BYTE_ARRAY that is the raw bytes with NO
// length prefix; for Decimal128 it is the 16 big-endian on-disk bytes, not
// the little-endian in-memory ones.
bool bloom_hash_row(const BoltColumn& col, BoltType t, std::int64_t r,
                    std::uint64_t* key) noexcept {
    assert(key != nullptr);
    if (t == BoltType::Utf8 || t == BoltType::Binary) {
        const RowVal v = row_val_bytes(col, r);
        if (v.n == 0xFFFFFFFFu) return false;
        *key = pq_xxh64(v.p, v.n, 0);
        return true;
    }
    if (t == BoltType::Decimal128) {
        std::uint8_t be[16];
        decimal_to_be(static_cast<const std::uint8_t*>(col.data) + r * 16, be);
        *key = pq_xxh64(be, 16u, 0);
        return true;
    }
    const std::size_t w = plain_elem_width(t);
    if (w == 0u) return false;
    const RowVal v = row_val_fixed(col, w, r);
    *key = pq_xxh64(v.p, v.n, 0);
    return true;
}

// Insert every non-null value of [0, n_rows).
bool bloom_build_chunk(const BoltColumn& col, BoltType t, std::int64_t n_rows,
                       const std::uint8_t* def_bits, bool nullable,
                       BloomBuilder* b) noexcept {
    assert(b != nullptr);
    assert(n_rows >= 0);
    for (std::int64_t r = 0; r < n_rows; ++r) {
        if (nullable && def_bits[r] == 0u) continue;
        std::uint64_t key = 0;
        if (!bloom_hash_row(col, t, r, &key)) return false;
        bloom_insert(b, key);
    }
    return true;
}

// BloomFilterHeader + the raw bitset. Each of the three unions is a struct
// carrying a single empty struct in field 1 -- BLOCK, XXHASH, UNCOMPRESSED
// respectively, the only combination the spec defines and the only one
// bolt's reader accepts.
void bloom_serialize(const BloomBuilder& b,
                     std::vector<std::uint8_t>* dst) noexcept {
    assert(dst != nullptr);
    assert(b.n_bytes >= 32u && b.bits.size() == b.n_bytes);
    dst->clear();
    TcOut o{dst};
    tc_put_field(&o, 1, kFI32);
    tc_put_zigzag(&o, static_cast<std::int64_t>(b.n_bytes));
    for (std::int16_t fid = 2; fid <= 4; ++fid) {
        tc_put_field(&o, fid, kFStruct);   // the union
        tc_put_field(&o, 1, kFStruct);     // arm 1
        tc_put_stop(&o);                   // empty arm struct
        tc_put_stop(&o);                   // end union
    }
    tc_put_stop(&o);
    dst->insert(dst->end(), b.bits.begin(), b.bits.end());
}
