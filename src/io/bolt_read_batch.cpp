// bolt_read_batch.cpp — range_coalesce and batched reads over a ReadPool;
// see bolt/io/bolt_file_io.h.

#include "bolt/io/bolt_file_io.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>

namespace bolt {
namespace io {

// --- range_coalesce ----------------------------------------------------------

namespace {

struct CoalesceSink {
    CoalescedRange* out;  // null: count only
    uint32_t cap;
    uint64_t n;           // requests produced (may exceed cap)
};

void emit(CoalesceSink* s, uint64_t off, uint64_t len, uint32_t first, uint32_t last) noexcept {
    assert(len != 0);
    assert(first <= last);
    if (s->out != nullptr && s->n < s->cap) s->out[s->n] = CoalescedRange{off, len, first, last - first + 1};
    ++s->n;
}

// Open request [s, e) built from inputs [first, last].
struct OpenReq {
    uint64_t s, e;
    uint32_t first, last;
    bool open;
};

// Start requests at lo for input i ending at end: whole max_request pieces,
// then leave the remainder open.
void start_at(CoalesceSink* sink, OpenReq* r, uint64_t lo, uint64_t end, uint64_t max_req,
              uint32_t i) noexcept {
    assert(lo < end);
    assert(!r->open);
    const uint64_t pieces = (end - lo - 1) / max_req;
    for (uint64_t k = 0; k < pieces; ++k, lo += max_req) emit(sink, lo, max_req, i, i);
    assert(end - lo <= max_req);
    *r = OpenReq{lo, end, i, i, true};
}

void coalesce_core(const IoRange* in, uint32_t n, uint64_t max_req, uint64_t max_gap,
                   CoalesceSink* sink) noexcept {
    assert(max_req > max_gap);
    OpenReq r{0, 0, 0, 0, false};
    for (uint32_t i = 0; i < n; ++i) {
        if (in[i].len == 0) continue;
        const uint64_t end = in[i].off + in[i].len;
        uint64_t lo = in[i].off;
        if (r.open) {
            if (end <= r.e) { r.last = i; continue; }  // already covered
            if (lo < r.e) lo = r.e;
            if (lo - r.e <= max_gap && end - r.s <= max_req) { r.e = end; r.last = i; continue; }
            emit(sink, r.s, r.e - r.s, r.first, r.last);
            r.open = false;
        }
        start_at(sink, &r, lo, end, max_req, i);
    }
    if (r.open) emit(sink, r.s, r.e - r.s, r.first, r.last);
}

bool coalesce_args_ok(const IoRange* in, uint32_t n, uint64_t max_req,
                      uint64_t max_gap) noexcept {
    if (max_req == 0 || max_req <= max_gap || (in == nullptr && n != 0)) return false;
    for (uint32_t i = 0; i < n; ++i) {
        if (in[i].len > UINT64_MAX - in[i].off) return false;
        if (i > 0 && in[i].off < in[i - 1].off) return false;
    }
    return true;
}

}  // namespace

FileIoStatus range_coalesce(const IoRange* in, uint32_t n, uint64_t max_request,
                            uint64_t max_gap, CoalescedRange* out, uint32_t out_cap,
                            uint32_t* out_n) noexcept {
    assert(out_n != nullptr);
    assert(out != nullptr || out_cap == 0);
    *out_n = 0;
    if (!coalesce_args_ok(in, n, max_request, max_gap)) return FileIoStatus::kBadArg;
    CoalesceSink sink{out, out_cap, 0};
    coalesce_core(in, n, max_request, max_gap, &sink);
    if (sink.n > UINT32_MAX) return FileIoStatus::kTooLarge;
    *out_n = static_cast<uint32_t>(sink.n);
    return sink.n <= out_cap ? FileIoStatus::kOk : FileIoStatus::kTooLarge;
}

uint32_t range_coalesce_count(const IoRange* in, uint32_t n, uint64_t max_request,
                              uint64_t max_gap) noexcept {
    if (!coalesce_args_ok(in, n, max_request, max_gap)) return 0;
    CoalesceSink sink{nullptr, 0, 0};
    coalesce_core(in, n, max_request, max_gap, &sink);
    assert(sink.out == nullptr);
    return sink.n > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(sink.n);
}

// --- batched reads -----------------------------------------------------------

struct ReadPool {
    std::mutex batch_mu;  // one batch at a time
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    uint64_t generation = 0;
    bool stop = false;
    uint32_t workers = 0;
    uint32_t busy = 0;    // workers still inside the current batch
    std::thread* threads = nullptr;
    const ReadFile* file = nullptr;
    ReadReq* reqs = nullptr;
    uint32_t n = 0;
    alignas(64) std::atomic<uint32_t> next{0};
};

namespace {

void run_one(const ReadFile& f, ReadReq* r) noexcept {
    assert(r != nullptr);
    r->got = 0;
    r->status = r->aligned ? file_pread_aligned(f, r->off, r->dst, r->len, &r->got)
                           : file_pread(f, r->off, r->dst, r->len, &r->got);
    assert(r->got <= r->len);
}

void drain(ReadPool* p) noexcept {
    assert(p->file != nullptr);
    for (uint32_t k = 0; k < p->n; ++k) {
        const uint32_t i = p->next.fetch_add(1, std::memory_order_relaxed);
        if (i >= p->n) return;
        run_one(*p->file, &p->reqs[i]);
    }
}

void worker_main(ReadPool* p) noexcept {
    assert(p != nullptr);
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv_work.wait(lk, [&] { return p->stop || p->generation != seen; });
            if (p->stop) return;
            seen = p->generation;
        }
        drain(p);
        std::lock_guard<std::mutex> lk(p->mu);
        assert(p->busy > 0);
        if (--p->busy == 0) p->cv_done.notify_one();
    }
}

}  // namespace

FileIoStatus read_pool_create(uint32_t workers, ReadPool** out) noexcept {
    assert(out != nullptr);
    *out = nullptr;
    if (workers > kReadPoolMaxWorkers) return FileIoStatus::kTooLarge;
    ReadPool* p = new (std::nothrow) ReadPool();
    if (p == nullptr) return FileIoStatus::kNoMemory;
    if (workers != 0) {
        p->threads = new (std::nothrow) std::thread[workers];
        if (p->threads == nullptr) { delete p; return FileIoStatus::kNoMemory; }
    }
    for (uint32_t i = 0; i < workers; ++i) {
        p->threads[i] = std::thread(worker_main, p);
        p->workers = i + 1;
    }
    assert(p->workers == workers);
    *out = p;
    return FileIoStatus::kOk;
}

void read_pool_destroy(ReadPool* p) noexcept {
    if (p == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->stop = true;
    }
    p->cv_work.notify_all();
    for (uint32_t i = 0; i < p->workers; ++i) p->threads[i].join();
    delete[] p->threads;
    delete p;
}

uint32_t read_pool_workers(const ReadPool* p) noexcept { return p == nullptr ? 0 : p->workers; }

namespace {

FileIoStatus first_error(const ReadReq* reqs, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i)
        if (reqs[i].status != FileIoStatus::kOk) return reqs[i].status;
    return FileIoStatus::kOk;
}

}  // namespace

FileIoStatus file_read_batch(ReadPool* p, const ReadFile& f, ReadReq* reqs,
                             uint32_t n) noexcept {
    assert(reqs != nullptr || n == 0);
    if (n > kReadBatchMaxRequests) return FileIoStatus::kTooLarge;
    if (p == nullptr || p->workers == 0 || n == 1) {
        for (uint32_t i = 0; i < n; ++i) run_one(f, &reqs[i]);
        return first_error(reqs, n);
    }
    std::lock_guard<std::mutex> batch(p->batch_mu);
    {
        std::lock_guard<std::mutex> lk(p->mu);
        assert(p->busy == 0);
        p->file = &f;
        p->reqs = reqs;
        p->n = n;
        p->next.store(0, std::memory_order_relaxed);
        p->busy = p->workers;
        ++p->generation;
    }
    p->cv_work.notify_all();
    drain(p);
    std::unique_lock<std::mutex> lk(p->mu);
    p->cv_done.wait(lk, [&] { return p->busy == 0; });
    p->file = nullptr;
    p->reqs = nullptr;
    return first_error(reqs, n);
}

}  // namespace io
}  // namespace bolt
