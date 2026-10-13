// Compare the former per-query property gather with indexed property weights.
// Graph/buffers/samples are allocated and touched before warmup. Timed work
// includes the former gather; output validation and sorting are outside it.
#include "bolt/kernels/bolt_csr_dijkstra_indexed.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace bk = bolt::kernels;
namespace {
constexpr uint32_t k_nodes = 10000, k_degree = 12, k_edges = k_nodes * k_degree;
constexpr uint32_t k_samples = 800, k_warmup = 100;
struct Fixture {
    std::vector<int64_t> off, values, scratch;
    std::vector<uint32_t> nbr, ids;
    std::vector<double> gathered, dist, reference;
    bk::CsrGraph32 graph{};
    Fixture() : off(k_nodes + 1), values(k_edges), scratch(2 * k_nodes),
        nbr(k_edges), ids(k_edges), gathered(k_edges), dist(k_nodes), reference(k_nodes) {
        for (uint32_t u = 0; u <= k_nodes; ++u) off[u] = u * k_degree;
        for (uint32_t i = 0; i < k_edges; ++i) {
            // A chain edge guarantees connectivity; other edges span the graph.
            const uint32_t u = i / k_degree, lane = i % k_degree;
            nbr[i] = (u + 1 + lane * 997) % k_nodes;
            ids[i] = (i * 7) % k_edges; // 7 is coprime to 120000.
            values[ids[i]] = 1 + static_cast<int64_t>((i * 13) % 97);
        }
        graph.off = off.data(); graph.neighbors = nbr.data();
        graph.edge_ids = ids.data(); graph.n_nodes = k_nodes; graph.want_label = -1;
    }
};

bool search(Fixture* f, int strategy, int64_t target, int64_t* settled) noexcept {
    assert(f != nullptr && settled != nullptr);
    assert(f->graph.n_nodes == k_nodes);
    if (strategy == 1) {
        const bk::CsrIndexedWeights<int64_t, uint32_t> weights{
            f->values.data(), nullptr, f->ids.data(), 0, k_edges};
        return bk::csr_dijkstra_indexed(&f->graph, weights, 0, target,
            f->dist.data(), nullptr, nullptr, f->scratch.data(), settled) == bk::CsrAlgoStatus::Ok;
    }
    if (strategy == 2) {
        const bk::CsrIndexedWeights<int64_t, uint32_t> weights{
            f->values.data(), nullptr, f->ids.data(), 0, k_edges};
        if (bk::csr_gather_indexed_weights(weights, k_edges, f->gathered.data(),
                                           k_edges) != bk::CsrAlgoStatus::Ok) return false;
    } else {
        for (uint32_t i = 0; i < k_edges; ++i) {
            const uint32_t id = f->ids[i];
            if (id >= k_edges) return false;
            f->gathered[i] = static_cast<double>(f->values[id]);
        }
    }
    return bk::csr_dijkstra(&f->graph, f->gathered.data(), 0, target,
        f->dist.data(), nullptr, nullptr, f->scratch.data(), settled) == bk::CsrAlgoStatus::Ok;
}

uint64_t now_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool measure(Fixture* f, int64_t target, int strategy) {
    assert(f != nullptr);
    assert(target == 0 || target == -1);
    std::vector<uint64_t> baseline(k_samples, 0), indexed(k_samples, 0);
    int64_t count = 0;
    if (!search(f, 0, target, &count)) return false;
    f->reference = f->dist;
    if (!search(f, strategy, target, &count) || f->dist != f->reference) return false;
    for (uint32_t i = 0; i < k_warmup; ++i)
        if (!search(f, (i & 1u) != 0 ? strategy : 0, target, &count)) return false;
    // Alternate order per sample to avoid placing every candidate after a
    // baseline warmed the same data. No allocations occur inside search.
    for (uint32_t i = 0; i < k_samples; ++i) {
        for (uint32_t arm = 0; arm < 2; ++arm) {
            const bool candidate = ((i + arm) & 1u) != 0;
            const uint64_t begin = now_ns();
            const bool ok = search(f, candidate ? strategy : 0, target, &count);
            const uint64_t elapsed = now_ns() - begin;
            if (!ok) return false;
            (candidate ? indexed : baseline)[i] = elapsed;
        }
    }
    if (f->dist != f->reference) return false;
    for (uint32_t i = 0; i < k_samples; ++i)
        std::printf("sample,%lld,%u,%llu,%llu\n", static_cast<long long>(target), i,
            static_cast<unsigned long long>(baseline[i]), static_cast<unsigned long long>(indexed[i]));
    for (int candidate = 0; candidate < 2; ++candidate) {
        auto& times = candidate ? indexed : baseline;
        std::sort(times.begin(), times.end());
        std::printf("summary,%lld,%s,p50_ns=%llu,p99_ns=%llu,settled=%lld\n",
            static_cast<long long>(target), candidate ?
                (strategy == 1 ? "indexed" : "typed_gather") : "materialized",
            static_cast<unsigned long long>(times[k_samples / 2]),
            static_cast<unsigned long long>(times[(k_samples * 99) / 100]),
            static_cast<long long>(count));
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const int strategy = argc == 1 ? 1 :
        (argc == 2 && std::strcmp(argv[1], "--typed-gather") == 0 ? 2 : 0);
    if (strategy == 0) return 2;
    Fixture fixture;
    std::puts("# single worker; resident synthetic graph; nodes=10000; edges=120000; warmup=100; samples=800; units=ns");
    return measure(&fixture, 0, strategy) && measure(&fixture, -1, strategy) ? 0 : 1;
}
