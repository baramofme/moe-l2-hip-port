#pragma once

#include <cstddef>

// [moe-l2 HIP] Expert staging engine (bounce buffers).
//
// HIP selective hipHostRegister crashes: hipMemcpyAsync H2D from an
// unregistered mmap page adjacent to a registered one fails with
// "invalid argument". The fix is to never register the model mmap and
// copy expert slices through pinned host buffers instead:
//
//   mmap -> (CPU memcpy) -> pinned staging buffer -> (async H2D) -> VRAM
//
// The engine keeps a small pool of double-buffered pinned allocations
// so the CPU memcpy of expert k+1 overlaps the GPU DMA of expert k.
// After an expert is safely stored in the VRAM LRU cache, its mmap pages
// are dropped from the page cache with madvise(MADV_PAGEOUT) to bound RSS.
//
// HIP-only. On CUDA builds these symbols do not exist, the proc-address
// lookups return nullptr, and the original selective-pin path is used.

// Copy src (model mmap, unpinned) to dst_gpu (VRAM) through a pinned
// staging buffer, on the given stream. Returns true on success.
// Larger-than-buffer experts are copied in chunks.
bool ggml_cuda_expert_staging_copy(const void * src, void * dst_gpu, size_t bytes, void * stream);

// Best-effort drop of the mmap pages [addr, addr+bytes) from the page
// cache. Page-aligned internally. No-op when MADV_PAGEOUT is unavailable.
void ggml_cuda_expert_staging_evict(const void * addr, size_t bytes);

// Best-effort prefetch of the mmap pages [addr, addr+bytes) into the page
// cache (MADV_WILLNEED). Used before an expert that was just evicted is
// accessed again, so the next copy hits resident pages instead of a cold
// page-fault (measured 1.7ms for 1.55MB). Page-aligned internally.
void ggml_cuda_expert_staging_prefetch(const void * addr, size_t bytes);

// Release all staging buffers and events. Safe to call even when uninitialized.
void ggml_cuda_expert_staging_free(void);