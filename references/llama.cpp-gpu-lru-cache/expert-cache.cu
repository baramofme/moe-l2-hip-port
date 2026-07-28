#include "expert-cache.cuh"
#include "common.cuh"

#include <cuda_runtime.h>
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
    int          expert_id; // expert index (for key matching)
    char *       dev_ptr;   // cached GPU buffer (owns the allocation)
    size_t       size;      // byte size of this slot's allocation
    uint64_t     timestamp; // LRU timestamp (larger = more recently used)
    bool         valid;     // slot is occupied
};

// ── cache state ────────────────────────────────────────────────
// Singleton: one cache per process.  In a multi-GPU setup each GPU
// gets its own device allocation inside the slot, but the LRU book-
// keeping is host-side and shared.

static struct {
    ExpertCacheSlot slots[EXPERT_CACHE_MAX_SLOTS];
    int             n_slots;         // active slot count (<= EXPERT_CACHE_MAX_SLOTS)
    int             device;          // CUDA device ordinal
    bool            initialized;     // cache has been set up
    uint64_t        clock;           // monotonically increasing timestamp
    size_t          default_expert_size; // per-expert byte size from init
} g_cache;

//
// Helpers
//

static void expert_cache_clear_slot(ExpertCacheSlot & slot) {
    if (slot.dev_ptr) {
        CUDA_CHECK(cudaFree(slot.dev_ptr));
    }
    slot.cpu_src   = nullptr;
    slot.expert_id = 0;
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

    // Clamp to 1.0 so the formula is safe
    if (fraction > 1.0f) {
        fraction = 1.0f;
    }

    int n_slots = (int)std::ceil(fraction * (float)n_expert);
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

    // size=0: no pre-allocation; each slot allocates on first miss.
    ggml_cuda_expert_cache_init(device, n_slots, 0);
}

void ggml_cuda_expert_cache_init(int device, int n_slots, size_t expert_size_bytes) {
    if (g_cache.initialized) {
        ggml_cuda_expert_cache_free();
    }

    int actual_slots = std::min(n_slots, EXPERT_CACHE_MAX_SLOTS);
    if (actual_slots <= 0) {
        actual_slots = 1;
    }

    // Pre-allocate device memory for each slot at init time.
    // This avoids per-token cudaMalloc which is slow.
    int prev_device;
    CUDA_CHECK(cudaGetDevice(&prev_device));
    CUDA_CHECK(cudaSetDevice(device));

    for (int i = 0; i < actual_slots; i++) {
        g_cache.slots[i].cpu_src   = nullptr;
        g_cache.slots[i].expert_id = 0;
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
            // If allocation fails, the slot stays null and will be
            // lazily allocated on first miss later.
        }
    }

    CUDA_CHECK(cudaSetDevice(prev_device));

    g_cache.n_slots              = actual_slots;
    g_cache.device               = device;
    g_cache.initialized          = true;
    g_cache.clock                = 0;
    g_cache.default_expert_size  = expert_size_bytes;
}

const void * ggml_cuda_expert_cache_lookup(
    int expert_id,
    const void * cpu_src,
    size_t expert_size,
    cudaStream_t stream)
{
    if (!g_cache.initialized) {
        return nullptr;
    }

    const uint64_t now = ++g_cache.clock;

    // ── 1. Linear scan for a hit ───────────────────────────────
    for (int i = 0; i < g_cache.n_slots; i++) {
        ExpertCacheSlot & slot = g_cache.slots[i];
        if (slot.valid &&
            slot.expert_id == expert_id &&
            slot.cpu_src   == cpu_src)
        {
            slot.timestamp = now;
            return slot.dev_ptr;
        }
    }

    // ── 2. Miss ── find LRU victim ────────────────────────────
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
        return nullptr; // shouldn't happen with n_slots > 0
    }

    ExpertCacheSlot & slot = g_cache.slots[victim];

    // ── 3. Evict old entry ─────────────────────────────────────
    if (slot.valid) {
        // Device memory is reused -- no cudaFree, just overwrite.
        slot.valid = false;
    }

    // ── 4. Ensure device buffer is large enough ────────────────
    if (slot.size < expert_size) {
        if (slot.dev_ptr) {
            CUDA_CHECK(cudaFree(slot.dev_ptr));
        }
        int prev_device;
        CUDA_CHECK(cudaGetDevice(&prev_device));
        CUDA_CHECK(cudaSetDevice(g_cache.device));

        void * ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, expert_size);
        CUDA_CHECK(cudaSetDevice(prev_device));

        if (err != cudaSuccess) {
            slot.dev_ptr = nullptr;
            slot.size    = 0;
            return nullptr;
        }
        slot.dev_ptr = (char *)ptr;
        slot.size    = expert_size;
    }

    // ── 5. Async H2D copy ─────────────────────────────────────
    CUDA_CHECK(cudaMemcpyAsync(
        slot.dev_ptr, cpu_src, expert_size,
        cudaMemcpyHostToDevice, stream));

    // ── 6. Fill slot metadata ──────────────────────────────────
    slot.cpu_src   = cpu_src;
    slot.expert_id = expert_id;
    slot.timestamp = now;
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
