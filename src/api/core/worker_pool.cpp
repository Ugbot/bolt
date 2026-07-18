// Worker Thread Pool Implementation
//
// Global pool singleton and initialization

#include "bolt/api/core/worker_pool.h"
#include <mutex>
#include <memory>

namespace bolt::api {
namespace core {

// Global pool singleton
static std::unique_ptr<WorkerThreadPool> g_worker_pool;
static std::once_flag g_worker_pool_init_flag;

WorkerThreadPool& global_worker_pool() {
    std::call_once(g_worker_pool_init_flag, []() {
        if (!g_worker_pool) {
            g_worker_pool = std::make_unique<WorkerThreadPool>(WorkerPoolConfig{});
            g_worker_pool->start();
        }
    });
    return *g_worker_pool;
}

void init_global_worker_pool(const WorkerPoolConfig& config) {
    std::call_once(g_worker_pool_init_flag, [&config]() {
        g_worker_pool = std::make_unique<WorkerThreadPool>(config);
        g_worker_pool->start();
    });
}

}  // namespace core
}  // namespace bolt::api
