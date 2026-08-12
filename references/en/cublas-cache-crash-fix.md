# llama.cpp expert_cache cuBLAS illegal memory access fix log

> **Historical record (2026-07-29)**: This document records the CUDA crash fix caused by repeatedly caching an oversized tensor (LM head 970 MB) in the A3 LRU expert cache under the **old architecture**.
> **Architecture upgraded since 2026-08-02** (host-buffer + cache mounted at the sched copy layer); this crash path no longer exists; the document is kept as a reference for troubleshooting cache crashes on other models.
> The benchmark data at the end (4.5-5.3 / 6.8-7.9 t/s) is from the old architecture; current performance is in [deepseek-v2-lite-q2k-benchmark.md](deepseek-v2-lite-q2k-benchmark.md) and [qwen3.6-a3b-iq2m-benchmark.md](qwen3.6-a3b-iq2m-benchmark.md).

## Background

Qwen3.6-35B-A3B-UD-IQ2_M crashes (exit 134) when `GGML_CUDA_EXPERT_CACHE>0`, stderr prints:
```
CUDA error 77: an illegal memory access was encountered
```
Trigger point: the last `force-cublas` call (cuBLAS gemm), tensor type=12 (Q4_K), ne0=2048 ne1=248320 (LM head output projection); the crash occurred on the last of 1074 force-cublas calls.

## Root cause

`cache_set` performed the following on the **970 MB LM head tensor** (Q4_K, 2048×248320):

```cpp
ggml_cuda_expert_cache_set(src0->data, cache_sz, src0_alloc.get(), main_stream);
```

Internally it does cudaMalloc 970 MB + cudaMemcpyAsync D2D; after allocation, on a cache miss the entry is evicted, and each trigger performs a 970 MB alloc → free → cuBLAS reading the same address. This LM head itself is evicted every 1-2 tokens (because it is computed per single token); the repeated large allocations cause CUDA memory fragmentation or page-table pollution, and eventually cuBLAS reports illegal memory access on sync.

Repeatedly copying the LM head is also worthless — it contains no expert weights; its only purpose would be to save one D2D copy when the next token happens to hit the LM head, but the LM head is just way too big.

## Fix

Added a size threshold check before the two `cache_set` calls to **skip tensors > 100 MB**:

### Location 1: cublas_impl (L1474)

```cpp
{   // Skip cache_set for huge tensors (> 100 MB) to avoid cuBLAS illegal memory access
    const size_t cache_sz = (size_t)ggml_nelements(src0) * sizeof(cuda_t);
    if (cache_sz <= 100 * 1024 * 1024) {
        ggml_cuda_expert_cache_set(src0->data, cache_sz, src0_alloc.get(), main_stream);
    }
}
```

### Location 2: mul_mat_id write-back path (L2075)

```cpp
// Write back to cache for next token (skip if > 100MB)
const void * cached = NULL;
if (nb02 <= 100ul * 1024 * 1024) {
    cached = ggml_cuda_expert_cache_set(
        cpu_src, nb02, temp_gpu.ptr, ctx.stream());
}
gpu_ptr = cached ? cached : temp_gpu.ptr;
```

## Verification

### Single smoke test
```
model=Qwen3.6-35B-A3B-UD-IQ2_M
cache=0.5  env GGML_CUDA_EXPERT_CACHE=0.5
Prompt: 8.7 t/s | Generation: 5.2 t/s  EXIT 0
```

cache_set call distribution after the fix (cache=0.5, single inference):
- 2148 cache_set calls
- of which 4 were LM head (970 MB) → **SKIPPED**
- 1070 small tensors (expert weights etc.) → cached normally, cumulative allocation ~17.45 GB

### Full benchmark (30 combos, all PASS)

| Model | Cache level | Conv type | Status | VRAM peak | Generation speed |
|------|-----------|----------|------|----------|---------|
| Qwen3.6-A3B IQ2_M | 0/0.1/0.5/1.0/2.0 | short/long/followup | **15/15 PASS** | 3.4→6.6 GB | 4.5-5.3 t/s |
| DS-V2-Lite Q2_K | 0/0.1/0.5/1.0/2.0 | short/long/followup | **15/15 PASS** | 1.7→3.4 GB | 6.8-7.9 t/s |

**0 failures across all 30 combos.**

## Why 100 MB is a safe threshold

- Only the LM head (970 MB) tensor was skipped
- All expert weight tensors are smaller than 100 MB and are cached normally
- The LM head is the only tensor that is both huge and frequently evicted — caching it has near-zero benefit and high side effects
- Other models with unusually huge single tensors can use the same logic to skip them

## Follow-up suggestions

If you hit expert_cache crashes on other models, the first thing to check is whether the largest tensor by `ggml_nelements * sizeof(cuda_t)` exceeds 100 MB. Large cudaMalloc allocations + frequent frees → fragmentation is a common root cause of CUDA illegal memory access.
