// Snappy + parquet decode benchmark.
//
// Two modes:
//   bench_parquet_decode                      codec only, self-contained
//   bench_parquet_decode <file.parquet> [n] [gb] [cols]   real parquet decode
//
// The codec mode needs NO data file. It synthesises a corpus, compresses it
// with a reference compressor written here, decompresses with bolt, and checks
// the result byte-for-byte before reporting throughput. That matters because
// bolt's own snappy_compress emits a single literal chunk by design (see
// bolt_snappy.h) and therefore cannot produce a back-reference at all — a
// round-trip through it would exercise none of the decoder's copy paths, which
// are where essentially all of the decode time goes.
//
// Timing is min-of-N. Decode is the only thing inside the timed region: the
// corpus build, the compression and the verification all sit outside it.
//
// The measured shape of real data, for anyone tuning against this (TPC-H SF10
// lineitem l_comment, 512 MB, 85.5M tags): 6.3 output bytes per tag, 86.7% of
// tags are copies with mean length 6.9, 99.3% of copies are len <= 16 with
// offset >= 8, and 0.11% overlap (offset < 8). Per-tag cost is the whole cost.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_column.h"
#include "bolt/ingest/bolt_snappy.h"
#include "bolt/ingest/bolt_parquet_read.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// A corpus with the redundancy structure of real text, from a fixed seed so
// every run and every machine measures the same bytes.
// The vocabulary size is a TUNING KNOB, not decoration: it sets the
// compression ratio, and the ratio sets output-bytes-per-tag, which is the
// thing the decoder's cost is proportional to. A small vocabulary compresses
// too well (a 24-word list gave ratio 3.86 against real data's 2.54) and
// flatters the decoder by giving it long matches and few tags.
//
// Sized to land near the measured ratio of real TPC-H l_comment. Run the codec
// mode and check the reported bytes-per-tag against the 6.3 recorded at the top
// of this file before trusting a number from it.
std::vector<uint8_t> make_corpus(uint64_t target) {
    static const char* kWords[] = {
        "the", "quick", "packages", "deposits", "furiously", "regular",
        "accounts", "express", "requests", "pending", "silent", "ironic",
        "carefully", "blithely", "unusual", "theodolites", "instructions",
        "final", "bold", "even", "slyly", "against", "according", "asymptotes",
        "dependencies", "excuses", "platelets", "warhorses", "dolphins",
        "attainments", "frets", "epitaphs", "somas", "dugouts", "hockey",
        "gifts", "sentiments", "notornis", "packets", "waters", "foxes",
        "courts", "dinos", "sauternes", "ideas", "asymptote", "braids",
        "multipliers", "escapades", "requests.", "accounts.", "deposits!",
        "packages?", "furiously,", "regular.", "pinto", "beans", "quickly",
        "special", "fluffily", "closely", "boldly", "busily", "doggedly"
    };
    constexpr uint32_t kNWords = sizeof(kWords) / sizeof(kWords[0]);
    std::vector<uint8_t> v;
    v.reserve(target + 64);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    while (v.size() < target) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;      // xorshift64
        // A minority of tokens carry digits, as real comment text does. This
        // is what keeps the tail of the vocabulary from repeating exactly and
        // holds the ratio down where real data sits.
        if (((s >> 7) & 15u) == 0u) {
            char num[12];
            const int n = std::snprintf(num, sizeof(num), "%u",
                                        static_cast<uint32_t>((s >> 17) % 99991u));
            v.insert(v.end(), num, num + n);
        } else {
            const char* w = kWords[(s >> 11) % kNWords];
            const size_t n = std::strlen(w);
            v.insert(v.end(), w, w + n);
        }
        v.push_back(' ');
    }
    v.resize(target);
    return v;
}

// ---------------------------------------------------------------------------
// Reference snappy compressor. Correctness and representative tag mix are what
// matter here, not compression speed or ratio -- it runs outside the timed
// region. It emits literals, copy-1 and copy-2, which is the mix real data has.
void emit_literal(std::vector<uint8_t>& out, const uint8_t* p, uint64_t n) {
    if (n == 0) return;
    const uint64_t v = n - 1;
    if (v < 60) {
        out.push_back(static_cast<uint8_t>(v << 2));
    } else {
        uint32_t nb = 1;
        while (nb < 4 && (v >> (8 * nb)) != 0) ++nb;
        out.push_back(static_cast<uint8_t>((59u + nb) << 2));
        for (uint32_t k = 0; k < nb; ++k) {
            out.push_back(static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
        }
    }
    out.insert(out.end(), p, p + n);
}

void emit_copy(std::vector<uint8_t>& out, uint64_t off, uint64_t len) {
    while (len > 0) {
        const uint64_t take = (len > 64) ? 64 : len;
        if (take >= 4 && take <= 11 && off < 2048) {          // copy-1
            out.push_back(static_cast<uint8_t>(((off >> 8) << 5) |
                                               ((take - 4) << 2) | 1u));
            out.push_back(static_cast<uint8_t>(off & 0xFF));
        } else {                                               // copy-2
            out.push_back(static_cast<uint8_t>(((take - 1) << 2) | 2u));
            out.push_back(static_cast<uint8_t>(off & 0xFF));
            out.push_back(static_cast<uint8_t>((off >> 8) & 0xFF));
        }
        len -= take;
    }
}

std::vector<uint8_t> compress_ref(const uint8_t* src, uint64_t n) {
    std::vector<uint8_t> out;
    out.reserve(n / 2 + 64);
    uint64_t v = n;                                   // uncompressed-length varint
    while (v >= 0x80) { out.push_back(uint8_t(v) | 0x80u); v >>= 7; }
    out.push_back(static_cast<uint8_t>(v));

    constexpr uint32_t kBits = 14;
    std::vector<uint32_t> table(size_t{1} << kBits, 0xFFFFFFFFu);
    uint64_t lit_start = 0, i = 0;
    while (i + 4 <= n) {
        uint32_t w;
        std::memcpy(&w, src + i, 4);
        const uint32_t h = (w * 0x1E35A7BDu) >> (32 - kBits);
        const uint32_t cand = table[h];
        table[h] = static_cast<uint32_t>(i);
        uint64_t off = 0;
        if (cand != 0xFFFFFFFFu && i > cand) {
            off = i - cand;
            uint32_t cw;
            std::memcpy(&cw, src + cand, 4);
            // 65535 is snappy's window; anything further needs copy-4, which
            // the reference compressor never emits (nor does Google's).
            if (cw != w || off == 0 || off > 65535) off = 0;
        }
        if (off == 0) { ++i; continue; }
        uint64_t len = 4;
        while (i + len < n && len < 64 && src[i + len] == src[i + len - off]) ++len;
        emit_literal(out, src + lit_start, i - lit_start);
        emit_copy(out, off, len);
        i += len;
        lit_start = i;
    }
    emit_literal(out, src + lit_start, n - lit_start);
    return out;
}

// ---------------------------------------------------------------------------
int run_codec(int passes) {
    constexpr uint64_t kChunk = 1024u * 1024u;        // ~ a parquet page
    constexpr uint64_t kTotal = 256u * 1024u * 1024u;
    std::fprintf(stderr, "building corpus (%llu MB)...\n",
                 (unsigned long long)(kTotal >> 20));
    const std::vector<uint8_t> raw = make_corpus(kTotal);

    std::vector<std::vector<uint8_t>> comp;
    uint64_t tot_c = 0;
    for (uint64_t o = 0; o < raw.size(); o += kChunk) {
        const uint64_t n = (raw.size() - o < kChunk) ? (raw.size() - o) : kChunk;
        comp.push_back(compress_ref(raw.data() + o, n));
        tot_c += comp.back().size();
    }
    // Report the shape, not just the ratio. Decode cost tracks TAGS, not bytes,
    // so bytes-per-tag is the number that says whether this corpus resembles
    // the workload. Walking the tag stream is the only honest way to get it.
    uint64_t n_lit = 0, n_copy = 0, copy_bytes = 0, copy_le16_off_ge8 = 0;
    for (const auto& c : comp) {
        uint64_t ip = 0;
        while (ip < c.size() && (c[ip] & 0x80u) != 0) ++ip;   // length varint
        ++ip;
        while (ip < c.size()) {
            const uint8_t tag = c[ip++];
            if ((tag & 3u) == 0u) {
                uint64_t len = (tag >> 2) + 1u;
                if (len > 60u) {
                    const uint32_t nb = static_cast<uint32_t>(len - 60u);
                    uint64_t val = 0;
                    for (uint32_t k = 0; k < nb; ++k) {
                        val |= static_cast<uint64_t>(c[ip + k]) << (8u * k);
                    }
                    ip += nb;
                    len = val + 1u;
                }
                ++n_lit;
                ip += len;
            } else {
                uint64_t len = 0, off = 0;
                if ((tag & 3u) == 1u) {
                    len = ((tag >> 2) & 7u) + 4u;
                    off = (static_cast<uint64_t>(tag >> 5) << 8) | c[ip];
                    ip += 1;
                } else {
                    len = (tag >> 2) + 1u;
                    off = static_cast<uint64_t>(c[ip]) |
                          (static_cast<uint64_t>(c[ip + 1]) << 8);
                    ip += 2;
                }
                ++n_copy;
                copy_bytes += len;
                if (len <= 16 && off >= 8) ++copy_le16_off_ge8;
            }
        }
    }
    const uint64_t tags = n_lit + n_copy;
    std::fprintf(stderr,
                 "%zu chunks, ratio %.2f, %llu tags, %.1f output bytes/tag\n"
                 "  copies %.1f%% of tags (mean %.1f B, %.1f%% are len<=16 "
                 "with off>=8)\n"
                 "  reference shape from real TPC-H l_comment: 6.3 bytes/tag, "
                 "86.7%% copies, mean 6.9 B, 99.3%%\n",
                 comp.size(), double(raw.size()) / double(tot_c),
                 (unsigned long long)tags, double(raw.size()) / double(tags),
                 100.0 * double(n_copy) / double(tags),
                 double(copy_bytes) / double(n_copy),
                 100.0 * double(copy_le16_off_ge8) / double(n_copy));

    std::vector<uint8_t> out(kChunk);
    // Correctness BEFORE throughput: a decoder that is fast and wrong is not a
    // faster decoder. Destination is sized exactly, with no slop.
    for (size_t c = 0; c < comp.size(); ++c) {
        const uint64_t o = uint64_t(c) * kChunk;
        const uint64_t n = (raw.size() - o < kChunk) ? (raw.size() - o) : kChunk;
        if (!bolt::ingest::snappy_decompress(comp[c].data(), comp[c].size(),
                                             out.data(), n)) {
            std::fprintf(stderr, "FAIL: chunk %zu did not decompress\n", c);
            return 1;
        }
        if (std::memcmp(out.data(), raw.data() + o, n) != 0) {
            std::fprintf(stderr, "FAIL: chunk %zu decoded wrong\n", c);
            return 1;
        }
    }
    std::fprintf(stderr, "round-trip verified byte-exact on every chunk\n");

    double best = 1e18;
    for (int p = 0; p < passes; ++p) {
        const auto t0 = Clock::now();
        for (size_t c = 0; c < comp.size(); ++c) {
            const uint64_t o = uint64_t(c) * kChunk;
            const uint64_t n = (raw.size() - o < kChunk) ? (raw.size() - o) : kChunk;
            bolt::ingest::snappy_decompress(comp[c].data(), comp[c].size(),
                                            out.data(), n);
        }
        const double ms = ms_since(t0);
        if (ms < best) best = ms;
        std::fprintf(stderr, "  pass %d: %8.1f ms\n", p, ms);
    }
    std::printf("snappy_decompress: %.1f ms  %.2f GB/s  (%.2f GB out, ratio %.2f)\n",
                best, double(raw.size()) / 1e9 / (best / 1000.0),
                double(raw.size()) / 1e9, double(raw.size()) / double(tot_c));
    return 0;
}

// ---------------------------------------------------------------------------
inline void mix(uint64_t& h, uint64_t x) noexcept {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
}

uint64_t width_of(bolt::BoltType t) noexcept {
    using T = bolt::BoltType;
    switch (t) {
        case T::Int64: case T::Float64: case T::Decimal64:
        case T::UInt64: case T::Timestamp: case T::Duration: return 8;
        case T::Int32: case T::Float32: case T::UInt32:
        case T::Date32: return 4;
        case T::Int16: case T::UInt16: return 2;
        case T::Int8:  case T::UInt8:  case T::Bool: return 1;
        default: return 0;                     // Utf8/Binary handled apart
    }
}

// Hash the VALUES, not the buffer address or any padding, so two builds agree
// only if they decoded the same data. A StringView of 12 bytes or fewer is
// inline; longer ones live at str_overflow_base + ref.offset.
uint64_t hash_col(const bolt::BoltColumn& c, int64_t rows) {
    uint64_t h = 0xcbf29ce484222325ull;
    mix(h, static_cast<uint64_t>(c.type));
    mix(h, static_cast<uint64_t>(rows));
    if (c.data == nullptr || rows <= 0) return h;
    if (c.type == bolt::BoltType::Utf8 || c.type == bolt::BoltType::Binary) {
        const auto* sv = static_cast<const bolt::StringView*>(c.data);
        for (int64_t r = 0; r < rows; ++r) {
            const uint32_t len = sv[r].length;
            const char* p = (len <= 12u)
                ? &sv[r].prefix[0]
                : static_cast<const char*>(c.str_overflow_base) + sv[r].ref.offset;
            mix(h, len);
            for (uint32_t k = 0; k < len; ++k) {
                mix(h, static_cast<uint8_t>(p[k]));
            }
        }
        return h;
    }
    const uint64_t w = width_of(c.type);
    if (w == 0) return h;
    const auto* b = static_cast<const uint8_t*>(c.data);
    for (int64_t r = 0; r < rows; ++r) {
        uint64_t v = 0;
        std::memcpy(&v, b + static_cast<uint64_t>(r) * w, w);
        mix(h, v);
    }
    return h;
}

int run_parquet(const char* path, int passes, uint64_t gb,
                const std::vector<uint16_t>& proj) {
    std::vector<uint8_t> buf;
    {
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) { std::fprintf(stderr, "open failed: %s\n", path); return 2; }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        buf.resize(static_cast<size_t>(n));
        const size_t got = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (got != buf.size()) { std::fprintf(stderr, "short read\n"); return 2; }
    }
    bolt::ArenaConfig mc; mc.initial_block_size = 16ull << 20;
                          mc.max_block_size     = 256ull << 20;
    bolt::ArenaConfig dc; dc.initial_block_size = 64ull << 20;
                          dc.max_block_size     = gb * (1ull << 30) / 32ull;
    bolt::Arena meta_arena(mc);
    bolt::Arena arena(dc);
    bolt::ingest::parquet::PqMeta meta{};
    if (!bolt::ingest::parquet::parquet_read_meta(buf.data(), buf.size(),
                                                  &meta_arena, &meta)) {
        std::fprintf(stderr, "parquet_read_meta failed\n");
        return 2;
    }
    std::fprintf(stderr, "%s: %u row groups, %u columns, %.2f GB\n", path,
                 meta.n_row_groups, meta.n_columns,
                 double(buf.size()) / double(1ull << 30));

    std::vector<bolt::BoltColumn> cols(meta.n_columns);
    double best = 1e18;
    uint64_t sum = 0, rows_total = 0;
    for (int p = 0; p < passes; ++p) {
        uint64_t h = 0x100000001b3ull, rows_seen = 0;
        double ms = 0.0;
        for (uint32_t g = 0; g < meta.n_row_groups; ++g) {
            int64_t rows = 0;
            const auto t0 = Clock::now();
            const bool ok =
                proj.empty()
                    ? bolt::ingest::parquet::parquet_read_row_group(
                          buf.data(), buf.size(), &meta, g, &arena,
                          cols.data(), &rows)
                    : bolt::ingest::parquet::parquet_read_row_group_cols(
                          buf.data(), buf.size(), &meta, g, proj.data(),
                          static_cast<uint32_t>(proj.size()), &arena,
                          cols.data(), &rows);
            ms += ms_since(t0);          // decode only; the hash is outside it
            if (!ok) { std::fprintf(stderr, "row group %u failed\n", g); return 2; }
            const uint32_t nc = proj.empty() ? meta.n_columns
                                             : static_cast<uint32_t>(proj.size());
            for (uint32_t c = 0; c < nc; ++c) h ^= hash_col(cols[c], rows);
            rows_seen += static_cast<uint64_t>(rows);
            arena.reset();               // bounded: one row group live
        }
        if (ms < best) best = ms;
        if (p == 0) { sum = h; rows_total = rows_seen; }
        else if (h != sum) {
            std::fprintf(stderr, "CHECKSUM UNSTABLE ACROSS PASSES\n");
            return 3;
        }
        std::fprintf(stderr, "  pass %d: %8.1f ms\n", p, ms);
    }
    // The checksum is the point: a change that speeds decode up must leave this
    // untouched, and identical decoded bytes cannot change any downstream result.
    std::printf("decode: %.1f ms  rows=%llu  checksum=%016llx\n", best,
                (unsigned long long)rows_total, (unsigned long long)sum);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return run_codec(3);
    if (std::strcmp(argv[1], "--codec") == 0) {
        return run_codec((argc > 2) ? std::atoi(argv[2]) : 3);
    }
    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        std::fprintf(stderr,
            "bench_parquet_decode                       codec only, no data file\n"
            "bench_parquet_decode --codec [passes]      codec, N passes\n"
            "bench_parquet_decode <file.parquet> [passes] [arena_gb] [cols_csv]\n");
        return 0;
    }
    const int passes = (argc > 2) ? std::atoi(argv[2]) : 3;
    const uint64_t gb = (argc > 3) ? static_cast<uint64_t>(std::atoi(argv[3])) : 12;
    std::vector<uint16_t> proj;
    if (argc > 4) {
        const char* q = argv[4];
        while (*q != '\0') {
            proj.push_back(static_cast<uint16_t>(std::atoi(q)));
            while (*q != '\0' && *q != ',') ++q;
            if (*q != '\0') ++q;
        }
    }
    return run_parquet(argv[1], passes, gb, proj);
}
