#pragma once

#include <cuda_runtime.h>
#include <cstddef>

// LRU Expert Cache for MoE weight offloading.
//
// Caches GPU copies of recently-used expert weight slices so that
// consecutive inference steps hitting the same experts can skip H2D
// transfers. The cache is host-managed (no GPU kernels) and uses a
// simple timestamp-based LRU eviction policy.
//
// Key: (cpu_source_address, expert_id) -- each (layer, expert) pair
// is a distinct cache entry even when two layers share an expert_id.
//
// The cache must be initialized before use and freed at context
// destruction (see ggml_cuda_free_data).

// Maximum number of cache slots (safety limit).
#define EXPERT_CACHE_MAX_SLOTS 64

// Lazy-init the expert cache from the GGML_CUDA_EXPERT_CACHE env var.
// Slot count is computed as a fraction of the model's per-layer expert
// count (n_expert), so the same setting works for any MoE model:
//   0, "", unset  → cache disabled (original behaviour)
//   0.25          → ceil(0.25 * n_expert) slots
//   0.5           → ceil(0.50 * n_expert) slots
//   0.75          → ceil(0.75 * n_expert) slots
//   1             → ceil(1.00 * n_expert) slots (cache all experts)
// n_expert is typically ne02 (number of experts in src0).
// Safe to call multiple times; only the first call with n_expert > 0
// has effect.
void ggml_cuda_expert_cache_maybe_init(int n_expert);

// Initialize the LRU expert cache for a given device.
//   device           - CUDA device ordinal
//   n_slots          - number of cache slots (clamped to EXPERT_CACHE_MAX_SLOTS)
//   expert_size_bytes- byte size of a single expert weight slice
// Returns 0 on success, -1 on error.
void ggml_cuda_expert_cache_init(int device, int n_slots, size_t expert_size_bytes);

// Look up an expert in the cache.  If the expert is present its cached
// GPU device pointer is returned.  If not present (miss), a new GPU
// buffer is allocated, the weight data is copied asynchronously from
// the CPU source via the given stream, inserted into the cache, and
// the new device pointer is returned.
//   expert_id  - expert index (i02 in the MUL_MAT_ID loop)
//   cpu_src    - CPU pointer to the expert weight data
//   expert_size- byte size of this expert's weight data
//   stream     - CUDA stream for the async H2D copy
// Returns a GPU device pointer, or NULL if the cache is not initialized.
const void * ggml_cuda_expert_cache_lookup(
    int expert_id,
    const void * cpu_src,
    size_t expert_size,
    cudaStream_t stream);

// Free all cached GPU buffers and reset the cache to uninitialized
// state. Safe to call even if the cache was never initialized.
void ggml_cuda_expert_cache_free(void);
