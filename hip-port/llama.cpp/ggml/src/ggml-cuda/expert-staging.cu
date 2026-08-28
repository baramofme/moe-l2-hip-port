#if defined(GGML_USE_HIP)

#include "common.cuh"
#include "expert-staging.h"

#include <atomic>
#include <array>
#include <cstring>
#include <cstdlib>
#include <list>
#include <mutex>
#include <new>
#include <unordered_set>
#include <unistd.h>
#include <sys/mman.h>

// Per-buffer-pair state. Two buffers alternate; the event of a buffer is
// waited on before that buffer is overwritten, so the CPU memcpy of the
// next expert overlaps the GPU DMA of the previous one on the same pair.
struct StagingPair {
    char *      buf[2] = {nullptr, nullptr};
    size_t      capacity = 0;
    cudaEvent_t ready[2] = {nullptr, nullptr};
    std::mutex  mtx;
    std::atomic<int> active{0};
};

static std::mutex g_init_mtx;
static std::array<StagingPair, 8> g_pairs;   // fixed max: clamp in staging_init
static std::atomic<unsigned> g_round_robin{0};
static int  g_n_pairs = 0;
static bool g_initialized = false;

// [moe-l2 HIP 2026-08-27] lazy init on first copy. Capacity follows the
// first expert size (clamped to a sane range); bigger experts chunk.
static void staging_init(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    if (g_initialized) {
        return;
    }

    const size_t ps = (size_t)getpagesize();
    const char * pairs_env = std::getenv("MOE_L2_STAGING_PAIRS");
    int n_pairs = pairs_env ? std::atoi(pairs_env) : 4;
    if (n_pairs < 1) n_pairs = 1;
    if (n_pairs > 8) n_pairs = 8;

    size_t cap = (bytes + ps - 1) & ~(ps - 1);
    if (cap < 2 * 1024 * 1024) cap = 2 * 1024 * 1024;
    if (cap > 32 * 1024 * 1024) cap = 32 * 1024 * 1024;

    g_n_pairs = n_pairs;
    for (int i = 0; i < n_pairs; i++) {
        StagingPair & p = g_pairs[i];
        for (int b = 0; b < 2; b++) {
            cudaMallocHost(&p.buf[b], cap);
            cudaEventCreateWithFlags(&p.ready[b], cudaEventDisableTiming);
        }
        p.capacity = cap;
    }
    g_initialized = true;
    fprintf(stderr, "[STAGING] init: %d pairs x 2 x %zu bytes (HIP bounce engine)\n",
            n_pairs, cap);
}

bool ggml_cuda_expert_staging_copy(const void * src, void * dst_gpu, size_t bytes, void * stream) {
    if (!src || !dst_gpu || bytes == 0) {
        return false;
    }
    if (!g_initialized) {
        staging_init(bytes);
    }

    cudaStream_t cs = (cudaStream_t)stream;
    StagingPair & p = g_pairs[g_round_robin.fetch_add(1, std::memory_order_relaxed) % (unsigned)g_n_pairs];
    std::lock_guard<std::mutex> lk(p.mtx);

    const int current = p.active.load(std::memory_order_relaxed);
    const int next    = 1 - current;

    // wait for the previous DMA from this buffer before overwriting it
    cudaEventSynchronize(p.ready[current]);

    const uint8_t * s = (const uint8_t *)src;
    uint8_t *       d = (uint8_t *)dst_gpu;
    size_t          remain = bytes;
    while (remain > 0) {
        const size_t chunk = remain < p.capacity ? remain : p.capacity;
        std::memcpy(p.buf[current], s, chunk);
        cudaMemcpyAsync(d, p.buf[current], chunk, cudaMemcpyHostToDevice, cs);
        s += chunk;
        d += chunk;
        remain -= chunk;
    }

    cudaEventRecord(p.ready[current], cs);
    p.active.store(next, std::memory_order_relaxed);
    return true;
}

// [moe-l2 HIP 2026-08-28] fixed-expert-count LRU (CUDA v3.1 port): keep at
// most MOE_L2_LRU_MAX_EXPERTS pages resident, evict only coldest overflow.
// Env unset -> legacy behavior: unconditional immediate evict.
static std::mutex g_res_mtx;
static std::list<std::pair<uintptr_t, size_t>> g_res;   // front = hottest
static std::unordered_set<uintptr_t> g_res_set;
static ssize_t g_res_max = -1;
static bool g_res_init = false;

static void resident_init() {
    if (g_res_init) {
        return;
    }
    g_res_init = true;
    const char * v = std::getenv("MOE_L2_LRU_MAX_EXPERTS");
    if (v && v[0] && std::atoi(v) > 0) {
        g_res_max = std::atoi(v);
    }
}

static void pagout(uintptr_t addr, size_t bytes) {
#if defined(MADV_PAGEOUT)
    const size_t ps = (size_t)getpagesize();
    uintptr_t start = addr & ~(uintptr_t)(ps - 1);
    uintptr_t end   = (addr + bytes + ps - 1) & ~(uintptr_t)(ps - 1);
    if (end > start) {
        madvise((void *)start, end - start, MADV_PAGEOUT);
    }
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(bytes);
#endif
}

void ggml_cuda_expert_staging_evict(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }
    resident_init();

    if (g_res_max < 0) {
        pagout((uintptr_t)addr, bytes);
        return;
    }

    std::lock_guard<std::mutex> lk(g_res_mtx);
    const uintptr_t key = (uintptr_t)addr;
    auto it = g_res_set.find(key);
    if (it != g_res_set.end()) {
        for (auto lit = g_res.begin(); lit != g_res.end(); ++lit) {
            if (lit->first == key) { g_res.erase(lit); break; }
        }
    } else {
        g_res_set.insert(key);
    }
    g_res.emplace_front(key, bytes);
    while ((ssize_t)g_res.size() > g_res_max) {
        auto cold = g_res.back();
        g_res.pop_back();
        g_res_set.erase(cold.first);
        pagout(cold.first, cold.second);
    }
}

void ggml_cuda_expert_staging_prefetch(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }
    const size_t ps = (size_t)getpagesize();
    uintptr_t start = (uintptr_t)addr & ~(uintptr_t)(ps - 1);
    uintptr_t end   = ((uintptr_t)addr + bytes + ps - 1) & ~(uintptr_t)(ps - 1);
    if (end <= start) {
        return;
    }
    madvise((void *)start, end - start, MADV_WILLNEED);
}

void ggml_cuda_expert_staging_free(void) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    if (!g_initialized) {
        return;
    }
    for (int i = 0; i < g_n_pairs; i++) {
        StagingPair & p = g_pairs[i];
        for (int b = 0; b < 2; b++) {
            if (p.buf[b]) {
                cudaFreeHost(p.buf[b]);
                p.buf[b] = nullptr;
            }
            if (p.ready[b]) {
                cudaEventDestroy(p.ready[b]);
                p.ready[b] = nullptr;
            }
        }
    }
    g_n_pairs = 0;
    g_initialized = false;
}

#endif // defined(GGML_USE_HIP)