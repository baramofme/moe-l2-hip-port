#include "expert-cache.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>

//
// Internal LRU cache structure
//

struct ExpertCacheSlot {
    const void * cpu_src;   // CPU base address (for key matching)
    char *       dev_ptr;   // cached GPU buffer (owns the allocation)
    size_t       size;      // byte size of this slot's allocation
    uint64_t     timestamp; // LRU timestamp (larger = more recently used)
    bool         valid;     // slot is occupied
};

// ── cache state ────────────────────────────────────────────────
// Singleton: one cache per process.

static struct {
    ExpertCacheSlot slots[EXPERT_CACHE_MAX_SLOTS];
    int             n_slots;         // active slot count (<= EXPERT_CACHE_MAX_SLOTS)
    int             device;          // CUDA device ordinal
    bool            initialized;     // cache has been set up
    uint64_t        clock;           // monotonically increasing timestamp
} g_cache;

// moe-l2: model layer count, set by ggml_cuda_expert_cache_set_n_layers().
// Slot formula uses it to cover the whole-model key space
// (n_layers x 3 expert tensors x n_expert). Default 1 = old behaviour.
static int g_n_layers = 1;

// [moe-l2 2026-08-13 DS fix] 多 stream 并发竞态：concurrent-stream 调度会
// 并行执行多个 copy_experts，无锁 g_cache 的 LRU victim 选择与 slot 写入
// 互相竞争 → slot 数据错乱 → hit 输出垃圾（DS 实测）。所有 cache 操作持锁。
static std::mutex g_cache_mtx;

//
// Helpers
//

static void expert_cache_clear_slot(ExpertCacheSlot & slot) {
    if (slot.dev_ptr) {
        CUDA_CHECK(cudaFree(slot.dev_ptr));
    }
    slot.cpu_src   = nullptr;
    slot.dev_ptr   = nullptr;
    slot.size      = 0;
    slot.timestamp = 0;
    slot.valid     = false;
}

//
// Public API
//

void ggml_cuda_expert_cache_maybe_init(int n_expert) {
    if (g_cache.initialized) {
        return;
    }
    if (n_expert <= 0) {
        return;
    }

    const char * env = std::getenv("GGML_CUDA_EXPERT_CACHE");
    if (!env || env[0] == '\0') {
        return;
    }

    float fraction = std::atof(env);
    if (fraction <= 0.0f) {
        return;
    }

    // moe-l2: layer count from env var set by llama.cpp at model load
    // (crosses DSO boundary reliably). setter API kept as fallback.
    const char * nl_env = std::getenv("MOE_L2_N_LAYERS");
    if (nl_env && nl_env[0] != '\0') {
        int nl = std::atoi(nl_env);
        if (nl > 0) {
            g_n_layers = nl;
        }
    }

    // moe-l2: allow up to 8x for large caches so hot experts
    // survive LRU pressure from the ~468 block accesses per token
    if (fraction > 8.0f) {
        fraction = 8.0f;
    }

    // [moe-l2 2026-08-13] if the CLI injected an exact slot count derived
    // from the selective-pin router map (MOE_L2_CACHE_SLOTS = n_layers *
    // top_k * 3 gate/up/down tensors), use it directly. The generic formula
    // below (fraction * n_expert * 3 * n_layers) computes e.g. 30720 slots
    // for Qwen but EXPERT_CACHE_MAX_SLOTS truncates it, and the truncated
    // 512-slot cache thrashed every token (measured hit rate 0.0%).
    const char * slots_env = std::getenv("MOE_L2_CACHE_SLOTS");
    int n_slots = 0;
    if (slots_env && slots_env[0] != '\0') {
        int ns = std::atoi(slots_env);
        if (ns > 0) {
            n_slots = ns;
        }
    }
    if (n_slots <= 0) {
        // fallback: scale slots by model layer count x 3 (gate/up/down
        // expert tensors per layer) so the cache covers the whole-model key
        // space. Old formula (fraction x n_expert) only covered ONE tensor's
        // experts, so with 32 layers x 3 tensors the effective hit rate ~0.
        const float scale = 3.0f * (float)g_n_layers;
        n_slots = (int)std::ceil(fraction * (float)n_expert * scale);
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (n_slots > EXPERT_CACHE_MAX_SLOTS) {
        n_slots = EXPERT_CACHE_MAX_SLOTS;
    }

    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return;
    }

    // size=0: no pre-allocation; each slot allocates on first set.
    ggml_cuda_expert_cache_init(device, n_slots, 0);
    fprintf(stderr, "[A3] expert_cache initialized: %d slots on device %d (fraction=%.2f, n_expert=%d)\n",
            n_slots, device, fraction, n_expert);
}

void ggml_cuda_expert_cache_set_n_layers(int n_layers) {
    if (n_layers > 0) {
        g_n_layers = n_layers;
    }
}

void ggml_cuda_expert_cache_init(int device, int n_slots, size_t expert_size_bytes) {
    if (g_cache.initialized) {
        ggml_cuda_expert_cache_free();
    }

    int actual_slots = std::min(n_slots, EXPERT_CACHE_MAX_SLOTS);
    if (actual_slots <= 0) {
        actual_slots = 1;
    }

    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(device));

    for (int i = 0; i < actual_slots; i++) {
        g_cache.slots[i].cpu_src   = nullptr;
        g_cache.slots[i].dev_ptr   = nullptr;
        g_cache.slots[i].size      = 0;
        g_cache.slots[i].timestamp = 0;
        g_cache.slots[i].valid     = false;

        if (expert_size_bytes > 0) {
            void * ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, expert_size_bytes);
            if (err == cudaSuccess) {
                g_cache.slots[i].dev_ptr = (char *)ptr;
                g_cache.slots[i].size    = expert_size_bytes;
            }
        }
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    g_cache.n_slots              = actual_slots;
    g_cache.device               = device;
    g_cache.initialized          = true;
    g_cache.clock                = 0;
}

const void * ggml_cuda_expert_cache_get(const void * cpu_src) {
    if (!g_cache.initialized || !cpu_src) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_cache_mtx);
    // Linear scan for a hit
    for (int i = 0; i < g_cache.n_slots; i++) {
        ExpertCacheSlot & slot = g_cache.slots[i];
        if (slot.valid && slot.cpu_src == cpu_src) {
            // Validate cached GPU pointer (may be stale across server requests)
            if (slot.dev_ptr) {
                // TEMP-DEBUG4: skip cudaPointerGetAttributes sync check to test
                // whether it is the ~3.1ms/mmid cost. Re-enable for server mode.
                slot.timestamp = ++g_cache.clock;
                return slot.dev_ptr;
            }
        }
    }
    return nullptr; // miss
}

bool ggml_cuda_expert_cache_copy_if_hit(
        const void * cache_key,
        void * dst_gpu, size_t dst_offset,
        size_t size, cudaStream_t stream) {
    if (!g_cache.initialized || !cache_key || !dst_gpu || size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_cache_mtx);

    const void * src = nullptr;
    for (int i = 0; i < g_cache.n_slots; i++) {
        ExpertCacheSlot & slot = g_cache.slots[i];
        if (slot.valid && slot.cpu_src == cache_key && slot.dev_ptr) {
            src = slot.dev_ptr;
            slot.timestamp = ++g_cache.clock;
            break;
        }
    }
    if (src == nullptr) {
        return false; // miss — caller falls back to CPU→GPU copy
    }

    // Hit: D2D copy from the cache slot into the destination GPU buffer.
    // The cache slot's device pointer is owned by the cache and stays valid
    // across requests (unlike sched input-copy buffers).
    {
        // [moe-l2 debug] log pointer types once to diagnose invalid-argument
        static int dbg = 0;
        if (dbg < 6) {
            cudaPointerAttributes pa_dst, pa_src;
            cudaError_t e_dst = cudaPointerGetAttributes(&pa_dst, (char *)dst_gpu + dst_offset);
            cudaError_t e_src = cudaPointerGetAttributes(&pa_src, src);
            fprintf(stderr, "[SCACHE-DBG] dst=%p dst_attr_err=%d type=%d src=%p src_err=%d type=%d size=%zu\n",
                (char *)dst_gpu + dst_offset, (int)e_dst, e_dst == cudaSuccess ? (int)pa_dst.type : -1,
                src, (int)e_src, e_src == cudaSuccess ? (int)pa_src.type : -1, size);
            dbg++;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        (char *)dst_gpu + dst_offset, src, size,
        cudaMemcpyDeviceToDevice, stream));
    return true;
}

const void * ggml_cuda_expert_cache_set(const void * cpu_src, size_t size, const void * dev_ptr, cudaStream_t stream) {
    if (!g_cache.initialized || !cpu_src || !dev_ptr || size == 0) {
        return nullptr;
    }

    // [moe-l2 2026-08-13 DS fix] victim 选择 + slot 写入全程持锁
    std::lock_guard<std::mutex> lock(g_cache_mtx);

    // ── Find LRU victim ────────────────────────────────────────
    int victim = -1;
    uint64_t oldest = UINT64_MAX;

    for (int i = 0; i < g_cache.n_slots; i++) {
        ExpertCacheSlot & slot = g_cache.slots[i];
        if (!slot.valid) {
            victim = i;
            break;
        }
        if (slot.timestamp < oldest) {
            oldest = slot.timestamp;
            victim = i;
        }
    }

    if (victim < 0) {
        return nullptr;
    }

    ExpertCacheSlot & slot = g_cache.slots[victim];

    // ── Evict old entry ────────────────────────────────────────
    if (slot.valid) {
        slot.cpu_src = nullptr;
        slot.valid = false;
    }

    // ── Ensure device buffer is large enough ────────────────────
    if (slot.size < size) {
        if (slot.dev_ptr) {
            CUDA_CHECK(cudaFree(slot.dev_ptr));
        }
        int prev_device;
        CUDA_CHECK(cudaGetDevice(&prev_device));
        CUDA_CHECK(cudaSetDevice(g_cache.device));

        void * ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        CUDA_CHECK(cudaSetDevice(prev_device));

        if (err != cudaSuccess) {
            slot.dev_ptr = nullptr;
            slot.size    = 0;
            return nullptr;
        }
        slot.dev_ptr = (char *)ptr;
        slot.size    = size;
    }

    // ── Async D2D copy (P0 fix v2 2026-08-13) ─────────────────
    // v1 (sync cudaMemcpy) fixed the race but serializes every miss.
    // v2: use the caller-provided stream (split_backend's stream, the SAME
    // stream that fills dst_gpu via H2D) so the D2D cache write is ordered
    // after the H2D fill without blocking. If stream is null (old callers)
    // fall back to sync copy for correctness.
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(
            slot.dev_ptr, dev_ptr, size,
            cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(
            slot.dev_ptr, dev_ptr, size,
            cudaMemcpyDeviceToDevice));
    }

    // ── Fill slot metadata ──────────────────────────────────────
    slot.cpu_src   = cpu_src;
    slot.timestamp = ++g_cache.clock;
    slot.valid     = true;

    return slot.dev_ptr;
}

void ggml_cuda_expert_cache_free(void) {
    if (!g_cache.initialized) {
        return;
    }

    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(g_cache.device));

    for (int i = 0; i < g_cache.n_slots; i++) {
        expert_cache_clear_slot(g_cache.slots[i]);
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    std::memset(&g_cache, 0, sizeof(g_cache));
}
