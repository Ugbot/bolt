// bolt_codec_parallel.h — encode many column chunks, one task per chunk
// (PROTOTYPE, see bolt_column_codec.h).
//
// Chunks are independent, so the unit of parallelism is the chunk. Each worker
// encodes into its own scratch arena (reset per chunk) and copies the result
// into the job's destination; nothing is shared between tasks. With no
// scheduler, one worker or one job the chunks run inline on the caller.

#pragma once

#include "bolt/bolt_scheduler.h"
#include "bolt/codec/bolt_column_codec.h"

namespace bolt {
namespace codec {

struct ChunkJob {
    Stream in;
    uint8_t* dst = nullptr;
    uint64_t cap = 0;
    uint64_t len = 0;
    bool ok = false;
};

// Conservative destination size for any chain (template worst case: a
// one-digit token per two bytes).
inline uint64_t encode_bound(const Stream& s) noexcept {
    if (s.kind == Kind::kBytes) return s.n + 64;
    if (s.kind == Kind::kInts) return s.n * 9 + 4096;
    return s.nbytes * 6 + s.n * 24 + 4096;
}

struct EncodeChunksCtx { ChunkJob* jobs; const Chain* chain; Arena** scratch; };

inline void encode_chunks_task(void* ud, uint32_t start, uint32_t end, uint32_t tid) noexcept {
    assert(ud != nullptr && start <= end);
    auto* ctx = static_cast<EncodeChunksCtx*>(ud);
    Arena& a = *ctx->scratch[tid];
    for (uint32_t j = start; j < end; ++j) {
        ChunkJob& job = ctx->jobs[j];
        a.reset();
        Buf out; out.a = &a;
        job.ok = encode_column(job.in, *ctx->chain, a, &out) && out.n <= job.cap;
        job.len = job.ok ? out.n : 0;
        if (job.ok) std::memcpy(job.dst, out.p, out.n);
    }
}

// scratch: one arena per scheduler worker (one is enough when sched is null).
inline bool encode_chunks(Scheduler* sched, ChunkJob* jobs, uint32_t n, const Chain& chain,
                          Arena** scratch) noexcept {
    assert((jobs != nullptr || n == 0) && scratch != nullptr && scratch[0] != nullptr);
    EncodeChunksCtx ctx{jobs, &chain, scratch};
    if (sched == nullptr || sched->thread_count() <= 1 || n <= 1) {
        encode_chunks_task(&ctx, 0, n, 0);
    } else {
        sched->submit_range(&encode_chunks_task, &ctx, n, 1);
        sched->wait_all();
    }
    bool ok = true;
    for (uint32_t j = 0; j < n; ++j) ok &= jobs[j].ok;
    return ok;
}

}  // namespace codec
}  // namespace bolt
