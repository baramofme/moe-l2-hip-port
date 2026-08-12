# A3 LRU Expert Cache Fix Log

> **Historical record (2026-07-29)**: this document records the fixes for three bugs in the A3 LRU expert cache under the **old architecture (`--cpu-moe`, experts computed on CPU)**.
> **Superseded by the host-buffer architecture since 2026-08-02**: experts go through the CUDA host buffer (CPU-pinned, zero VRAM) + direct GPU compute, and the cache now lives in the sched copy layer (see [cache-sched-layer-benchmark.md](cache-sched-layer-benchmark.md)).
> The data below reflects only the old architecture and no longer represents current performance.

## Bugs found (3)

### 1. `--expert-cache` CLI flag never reached the backend
- `common/arg.cpp:2260` stores the value into `params.expert_cache_fraction`
- **It was never converted into an environment variable**; the backend `expert_cache_maybe_init` reads `GGML_CUDA_EXPERT_CACHE`
- `GGML_CUDA_EXPERT_CACHE=1` must be set explicitly

### 2. Cache key incorrectly included expert_id
- The slot structure used `(cpu_src, expert_id)` as the key
- When the cache was consulted in `mul_mat`, `expert_id` was hardcoded to `0`
- All 128 experts overwrote each other; only one slot was ever cached
- **Fix**: keep only `cpu_src` in the key

### 3. DS-V2-Lite took the mmvq path, so the cache code never ran
- In `ggml_cuda_mul_mat()`, single-token decode goes through the **mmvq** kernel (quantized × vector)
- The cache code lived in `cublas_impl` and **was never reached**
- **Fix**: insert a forced cublas path before the mmvq check (skip mmvq/mmq when `GGML_CUDA_EXPERT_CACHE` is set and the type is quantized)

## Measurements after the fix

| Metric | No cache | With cache | Change |
|------|--------|--------|------|
| VRAM | 1275 MiB | **4895 MiB** | ✅ +3.6 GiB |
| Gen | 8.6 t/s | 7.2 t/s | ↓ slightly slower |

**Conclusion**: the cache code now works correctly (128 experts × weights allocated). But mmvq is already optimal for single tokens, and forcing cublas is actually slightly slower. The cache is expected to be more valuable for batched inference.

## Patched files

- `ggml-cuda.cu` — lines 1855-1863 of `ggml_cuda_mul_mat()`, forced cublas path
- `expert-cache.cu` — key simplified to `cpu_src` only
