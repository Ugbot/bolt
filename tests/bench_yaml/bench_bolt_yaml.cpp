// bench_bolt_yaml — YAML differential CLI and microbenchmark (G2GRAPH-453).
//
//   bench_bolt_yaml --json FILE         one document -> compact JSON on stdout,
//                                       or `ERROR <code> <line> <what>` (exit 2)
//   bench_bolt_yaml --bench CORPUS [SECONDS]
//       CORPUS holds documents separated by NUL bytes. Prints one JSON line of
//       MB/s for: the tape build (build_index), to_json over the tape, the
//       previous text emitter (legacy to_json), and rapidyaml when built with
//       BOLT_BENCH_RYML.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bolt/bolt_arena.h"
#include "bolt/parse/bolt_yaml.h"
#include "yaml_legacy_to_json.inc"

#if defined(BOLT_BENCH_RYML)
#define RYML_SINGLE_HDR_DEFINE_NOW
#include BOLT_BENCH_RYML
#endif

namespace y = bolt::parse::yaml;

namespace {

constexpr y::YamlLimits kLim{1u << 30, 64u, 1u << 26};

std::string read_all(const char* path) {
    std::string s;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return s;
    char buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) s.append(buf, n);
    std::fclose(f);
    return s;
}

const char* code_name(y::YamlCode c) {
    switch (c) {
    case y::YamlCode::Ok: return "ok";
    case y::YamlCode::Syntax: return "syntax";
    case y::YamlCode::Unsupported: return "unsupported";
    case y::YamlCode::Limit: return "limit";
    case y::YamlCode::NoMemory: return "nomemory";
    }
    return "?";
}

int one(const char* path) {
    const std::string s = read_all(path);
    bolt::Arena arena;
    const char* out = nullptr;
    uint32_t n = 0;
    y::YamlError err{};
    const y::YamlCode c = y::to_json(s.data(), static_cast<uint32_t>(s.size()), &arena, kLim, &out, &n, &err);
    if (c != y::YamlCode::Ok) {
        std::printf("ERROR %s %u %s\n", code_name(c), err.line, err.what);
        return 2;
    }
    std::fwrite(out, 1, n, stdout);
    std::printf("\n");
    return 0;
}

template <typename Fn>
double mbps(const std::vector<std::string>& docs, size_t bytes, double seconds, Fn&& fn) {
    using clock = std::chrono::steady_clock;
    size_t done = 0;
    const auto t0 = clock::now();
    double el = 0;
    do {
        for (const std::string& d : docs) fn(d);
        done += bytes;
        el = std::chrono::duration<double>(clock::now() - t0).count();
    } while (el < seconds);
    return static_cast<double>(done) / el / 1e6;
}

int bench(const char* path, double seconds) {
    const std::string all = read_all(path);
    std::vector<std::string> docs;
    size_t bytes = 0;
    for (size_t b = 0; b < all.size();) {
        size_t e = all.find('\0', b);
        if (e == std::string::npos) e = all.size();
        docs.emplace_back(all.substr(b, e - b));
        bytes += e - b;
        b = e + 1;
    }
    bolt::Arena arena;
    size_t failures = 0;
    const double tape = mbps(docs, bytes, seconds, [&](const std::string& d) {
        arena.reset();
        bolt::parse::json::StructuralIndex idx{};
        y::YamlError err{};
        failures += y::build_index(d.data(), static_cast<uint32_t>(d.size()), &arena, kLim, nullptr, &idx, &err) != y::YamlCode::Ok;
    });
    auto text = [&](auto fn) {
        return mbps(docs, bytes, seconds, [&](const std::string& d) {
            arena.reset();
            const char* out = nullptr;
            uint32_t n = 0;
            y::YamlError err{};
            failures += fn(d.data(), static_cast<uint32_t>(d.size()), &arena, kLim, &out, &n, &err) != y::YamlCode::Ok;
        });
    };
    const double json_new = text(y::to_json);
    const double json_legacy = text(bolt::parse::yaml_legacy::to_json);
    double ryml = -1;
#if defined(BOLT_BENCH_RYML)
    ryml::Tree tree;
    std::string scratch;
    ryml = mbps(docs, bytes, seconds, [&](const std::string& d) {
        scratch = d;
        tree.clear();
        tree.clear_arena();
        ryml::parse_in_place(ryml::substr(scratch.data(), scratch.size()), &tree);
    });
#endif
    std::printf("{\"docs\":%zu,\"bytes\":%zu,\"failures\":%zu,\"tape_build_mb_s\":%.1f,\"to_json_tape_mb_s\":%.1f,"
                "\"to_json_legacy_mb_s\":%.1f,\"rapidyaml_tree_mb_s\":%.1f}\n",
                docs.size(), bytes, failures, tape, json_new, json_legacy, ryml);
    return failures == 0 ? 0 : 1;
}

}  // namespace

// One JSON (or ERROR) line per NUL-separated document.
int lines(const char* path) {
    const std::string all = read_all(path);
    bolt::Arena arena;
    for (size_t b = 0; b <= all.size();) {
        size_t e = all.find('\0', b);
        if (e == std::string::npos) e = all.size();
        arena.reset();
        const char* out = nullptr;
        uint32_t n = 0;
        y::YamlError err{};
        const y::YamlCode c = y::to_json(all.data() + b, static_cast<uint32_t>(e - b), &arena, kLim, &out, &n, &err);
        if (c == y::YamlCode::Ok) { std::fwrite(out, 1, n, stdout); std::printf("\n"); }
        else std::printf("ERROR %s %u %s\n", code_name(c), err.line, err.what);
        if (e == all.size()) break;
        b = e + 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--json") == 0) return one(argv[2]);
    if (argc >= 3 && std::strcmp(argv[1], "--jsonl") == 0) return lines(argv[2]);
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0) return bench(argv[2], argc >= 4 ? std::atof(argv[3]) : 1.0);
    std::fprintf(stderr, "usage: bench_bolt_yaml --json FILE | --bench CORPUS [SECONDS]\n");
    return 64;
}
