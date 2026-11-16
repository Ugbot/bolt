// bench_column_codec.cpp — column codec chains over a real text / NDJSON file:
// bytes per row, ratio, encode and decode MB/s, every chunk round-trip checked.
//
//   bench_column_codec <file> [--rows N] [--chunk R] [--reps K] [--threads T] [--col NAME]
//                      [--str "chain;chain"] [--int "chain;chain"]
//
// Every line is one row of column `line`. If the lines are flat JSON objects,
// each key also becomes a column (a canonical integer value -> int64, else
// string). The column is cut into chunks of R rows (default 65536); each chunk
// is encoded on its own, T chunks at a time (default 1, inline); decode is 1
// thread. MB/s is raw column bytes over the best rep.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_scheduler.h"
#include "bolt/codec/bolt_codec_parallel.h"

namespace {

namespace C = bolt::codec;
using Clock = std::chrono::steady_clock;

struct Column {
    std::string name;
    bool is_int = true;
    std::vector<int64_t> ints;
    std::vector<uint32_t> offs{0};
    std::vector<uint8_t> bytes;
    std::vector<std::string> raw;   // values until the type is settled
};

bool canon_int(const std::string& s, int64_t* v) {
    if (s.empty() || s.size() > 18 || (s.size() > 1 && s[0] == '0')) return false;
    int64_t x = 0;
    for (char c : s) { if (c < '0' || c > '9') return false; x = x * 10 + (c - '0'); }
    *v = x; return true;
}

// Flat one-level JSON object, no escapes handled beyond \" inside strings.
bool json_fields(const std::string& line, std::vector<std::pair<std::string, std::string>>* out) {
    out->clear();
    size_t i = line.find('{');
    if (i == std::string::npos) return false;
    for (++i;;) {
        while (i < line.size() && (line[i] == ' ' || line[i] == ',')) ++i;
        if (i >= line.size()) return false;
        if (line[i] == '}') return !out->empty();
        if (line[i] != '"') return false;
        const size_t ke = line.find('"', i + 1);
        if (ke == std::string::npos) return false;
        std::string key = line.substr(i + 1, ke - i - 1);
        i = line.find(':', ke);
        if (i == std::string::npos) return false;
        ++i; while (i < line.size() && line[i] == ' ') ++i;
        std::string val;
        if (i < line.size() && line[i] == '"') {
            size_t e = i + 1;
            while (e < line.size() && (line[e] != '"' || line[e - 1] == '\\')) ++e;
            if (e >= line.size()) return false;
            val = line.substr(i + 1, e - i - 1); i = e + 1;
        } else {
            size_t e = i;
            while (e < line.size() && line[e] != ',' && line[e] != '}') ++e;
            val = line.substr(i, e - i);
            while (!val.empty() && val.back() == ' ') val.pop_back();
            i = e;
            if (val.empty() || val[0] == '{' || val[0] == '[') return false;
        }
        out->emplace_back(std::move(key), std::move(val));
    }
}

void settle(Column& c) {
    int64_t v;
    for (const auto& s : c.raw) if (!canon_int(s, &v)) { c.is_int = false; break; }
    for (const auto& s : c.raw) {
        if (c.is_int) { canon_int(s, &v); c.ints.push_back(v); }
        else { c.bytes.insert(c.bytes.end(), s.begin(), s.end()); c.offs.push_back(static_cast<uint32_t>(c.bytes.size())); }
    }
    c.raw.clear(); c.raw.shrink_to_fit();
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; size_t at = 0;
    for (;;) {
        const size_t e = s.find(';', at);
        v.push_back(s.substr(at, e == std::string::npos ? e : e - at));
        if (e == std::string::npos) return v;
        at = e + 1;
    }
}

struct Result { uint64_t enc_bytes = 0; double enc_s = 1e30, dec_s = 1e30; bool ok = true; };

Result run(const Column& c, const C::Chain& chain, uint64_t chunk, int reps, bolt::Scheduler* sched,
           bolt::Arena** scratch) {
    Result best;
    const uint64_t n = c.is_int ? c.ints.size() : c.offs.size() - 1;
    const uint64_t nchunks = (n + chunk - 1) / chunk;
    std::vector<std::vector<uint32_t>> local(nchunks);
    std::vector<std::vector<uint8_t>> dst(nchunks);
    std::vector<C::ChunkJob> jobs(nchunks);
    for (uint64_t j = 0; j < nchunks; ++j) {
        const uint64_t at = j * chunk, m = std::min(chunk, n - at);
        if (c.is_int) jobs[j].in = C::ints_stream(c.ints.data() + at, m);
        else {
            local[j].resize(m + 1);
            for (uint64_t i = 0; i <= m; ++i) local[j][i] = c.offs[at + i] - c.offs[at];
            jobs[j].in = C::strs_stream(local[j].data(), c.bytes.data() + c.offs[at], m);
        }
        dst[j].resize(C::encode_bound(jobs[j].in));
        jobs[j].dst = dst[j].data(); jobs[j].cap = dst[j].size();
    }
    bolt::Arena a;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = Clock::now();
        const bool eok = C::encode_chunks(sched, jobs.data(), static_cast<uint32_t>(nchunks), chain, scratch);
        const double es = std::chrono::duration<double>(Clock::now() - t0).count();
        if (!eok) { best.ok = false; return best; }
        uint64_t total = 0; double ds = 0;
        for (uint64_t j = 0; j < nchunks; ++j) {
            const C::Stream& in = jobs[j].in;
            total += jobs[j].len;
            a.reset();
            C::Stream back;
            const auto t1 = Clock::now();
            const bool dok = C::decode_column(jobs[j].dst, jobs[j].len, a, &back);
            ds += std::chrono::duration<double>(Clock::now() - t1).count();
            bool same = dok && back.n == in.n;
            if (same && c.is_int) same = std::memcmp(back.ints, in.ints, in.n * 8) == 0;
            if (same && !c.is_int)
                same = std::memcmp(back.offs, in.offs, (in.n + 1) * 4) == 0 &&
                       std::memcmp(back.bytes, in.bytes, in.nbytes) == 0;
            if (!same) { best.ok = false; return best; }
        }
        best.enc_bytes = total; best.enc_s = std::min(best.enc_s, es); best.dec_s = std::min(best.dec_s, ds);
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_column_codec <file> [--rows N] [--chunk R] [--reps K] [--str chains] [--int chains]\n"); return 2; }
    uint64_t max_rows = ~0ull, chunk = 65536; int reps = 3; uint32_t threads = 1; std::string only;
    std::string str_chains = "none;lz4;zstd;zstd(9);zstdself;dict,for,zstd;front,for,zstd;template,dict,for;"
                             "template,dict,for,lz4;template,dict,for,zstd;template(2),dict,for,zstd;"
                             "template(1),dict,for,zstd;template,dict,front,delta,for,zstd";
    std::string int_chains = "none;for;delta,for;rle,for;rle,delta,for;zstd;delta,for,zstd;rle,delta,for,zstd";
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--rows") max_rows = std::strtoull(argv[i + 1], nullptr, 10);
        else if (k == "--chunk") chunk = std::strtoull(argv[i + 1], nullptr, 10);
        else if (k == "--reps") reps = std::atoi(argv[i + 1]);
        else if (k == "--threads") threads = static_cast<uint32_t>(std::atoi(argv[i + 1]));
        else if (k == "--col") only = argv[i + 1];
        else if (k == "--str") str_chains = argv[i + 1];
        else if (k == "--int") int_chains = argv[i + 1];
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (chunk == 0 || chunk > C::kMaxRows || reps < 1 || threads < 1 || threads > 256) { std::fprintf(stderr, "bad --chunk / --reps\n"); return 2; }

    std::ifstream f(argv[1]);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::vector<Column> cols(1); cols[0].name = "line"; cols[0].is_int = false;
    std::map<std::string, size_t> by_name;
    std::vector<std::pair<std::string, std::string>> kv;
    std::string line; uint64_t rows = 0, raw_bytes = 0; bool json = true;
    while (rows < max_rows && std::getline(f, line)) {
        raw_bytes += line.size() + 1;
        if (json && json_fields(line, &kv)) {
            for (auto& p : kv) {
                auto it = by_name.find(p.first);
                if (it == by_name.end()) {
                    if (rows) { json = false; break; }
                    it = by_name.emplace(p.first, cols.size()).first;
                    cols.emplace_back(); cols.back().name = p.first;
                }
                cols[it->second].raw.push_back(std::move(p.second));
            }
        } else json = false;
        cols[0].raw.push_back(line); ++rows;
    }
    if (!json) cols.resize(1);
    for (auto& c : cols) { if (c.raw.size() != rows) { cols.resize(1); break; } }
    cols[0].is_int = false;
    for (size_t i = 0; i < cols.size(); ++i) { if (i) cols[i].is_int = true; settle(cols[i]); }

    bolt::Scheduler sched;
    if (threads > 1 && !sched.init(threads)) { std::fprintf(stderr, "scheduler init failed\n"); return 1; }
    std::vector<bolt::Arena> arenas(threads);
    std::vector<bolt::Arena*> scratch;
    for (auto& ar : arenas) scratch.push_back(&ar);
    std::printf("file %s: %llu rows, %.1f raw bytes/row, chunk %llu rows, %u encode thread(s), decode 1 thread, libzstd %s\n\n", argv[1],
                static_cast<unsigned long long>(rows), rows ? double(raw_bytes) / double(rows) : 0.0,
                static_cast<unsigned long long>(chunk), threads, bolt::ingest::zstd_available() ? "yes" : "no (zstd = bolt's own encoder)");
    std::printf("| column | chain | bytes/row | ratio | enc MB/s | dec MB/s |\n|---|---|---|---|---|---|\n");
    for (const auto& c : cols) {
        if (!only.empty() && c.name != only) continue;
        const double col_bytes = c.is_int ? double(c.ints.size()) * 8 : double(c.bytes.size());
        for (const auto& text : split(c.is_int ? int_chains : str_chains)) {
            C::Chain chain;
            if (!C::chain_parse(text.c_str(), &chain)) { std::fprintf(stderr, "bad chain %s\n", text.c_str()); return 2; }
            const Result r = run(c, chain, chunk, reps, threads > 1 ? &sched : nullptr, scratch.data());
            if (!r.ok) { std::fprintf(stderr, "ROUND TRIP FAILED: %s %s\n", c.name.c_str(), text.c_str()); return 1; }
            std::printf("| %s (%s, %.1f B/row raw) | %s | %.2f | %.1fx | %.0f | %.0f |\n", c.name.c_str(),
                        c.is_int ? "int64" : "string", rows ? col_bytes / double(rows) : 0.0, text.c_str(),
                        double(r.enc_bytes) / double(rows), col_bytes / double(r.enc_bytes),
                        col_bytes / 1e6 / r.enc_s, col_bytes / 1e6 / r.dec_s);
        }
    }
    if (threads > 1) sched.shutdown();
    return 0;
}
