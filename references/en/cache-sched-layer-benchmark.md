# moe-l2 cache on sched copy layer benchmark (2026-08-02)

## Background

The A3 LRU expert cache was originally mounted at the mul_mat_id **compute layer**, but expert copies happen at the sched **input copy layer** (earlier and lower-level), so the compute layer never sees CPU pointers → the cache was dead code. This time we mount the cache at the sched copy layer: in `ggml-backend.cpp`, `copy_experts` checks the cache before copying — on hit → D2D, no PCIe round trip; on miss → original CPU copy + write-back.

## Implementation

- `expert-cache.cuh/cu`: added `ggml_cuda_expert_cache_copy_if_hit()` (on hit, D2D copy into the target GPU buffer)
- `ggml-backend.cpp`: `copy_experts` routes single-expert groups through the cache (proc-address cross-DSO call; multi-expert groups fall back to the original path)
- `ggml-cuda.cu`: `maybe_init` moved up to the mul_mat_id function entry (the fast path also initializes the cache) + registry exposes the cache functions

## Three-model validation (RTX 4090, host buffer + OFFLOAD_MIN_BATCH=1)

### DS-V2-Lite Q2_K (expert 1.55MB, top-6)

| cache | Prompt t/s | Gen t/s | VRAM | Crash |
|-------|-----------|---------|------|------|
| none | 99.0 | 37.4 | 1625 MiB | 0 |
| **0.25 (optimal)** | **308.4 (+211%)** | **39.2 (+5%)** | 1625 | 0 |
| 0.5 | 308.8 | 39.4 | 2127 (+502) | 0 |
| 0.75 | 303.3 | 39.5 | 1625 | 0 |
| 1.0 | 304.2 | 39.4 | 2165 (+540) | 0 |

### Qwen3.6-35B-A3B IQ2_M (expert ~1MB, top-8)

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| none | 75.8 | 46.5 | 2147 MiB |
| 0.25 | 76.0 | 46.6 | 2147 MiB |
| 0.5 | 75.6 | 46.5 | 2475 MiB |

### Mixtral-8x7B Q4_K_M (expert ~252MB, top-2)

| cache | Prompt t/s | Gen t/s | VRAM |
|-------|-----------|---------|------|
| none | 15.0 | 3.7 | 2243 MiB |
| 0.25 | 15.1 | 3.7 | 2903 (+660) |
| 0.5 | 15.1 | 3.7 | 2903 (+660) |

## Benefit pattern

**cache benefit = expert size × hit rate**:

| Model | Expert size | Activation | Benefit |
|------|---------|---------|------|
| DS-V2-Lite | 1.55 MB | top-6 | **Prompt +211%, Gen +5%** |
| Qwen3.6-A3B | ~1 MB | top-8 | none (experts too small, moving them was never expensive) |
| Mixtral-8x7B | 252 MB | top-2 | none (top-2 hit rate too low, and slots consume lots of VRAM) |

## Tier conclusions

On DS, cache=0.25 is already at the ceiling (16 slots/layer covers all hot experts); 0.5/0.75/1.0 are all flat (303-309 / 39.4-39.5); larger tiers only add VRAM (+500MiB) with no speed gain.

## Recommended config (per model)

| Model | Recommendation | Reason |
|------|------|------|
| DS-V2-Lite | `GGML_CUDA_EXPERT_CACHE=0.25` | Prompt +211%, zero VRAM increase |
| Qwen3.6-A3B | cache off | experts too small, no benefit |
| Mixtral-8x7B | cache off | top-2 hit rate low, wastes VRAM |

## Pitfalls log

1. **Cache slots are allocated per single expert**: consecutive multi-expert groups (first_id≠last_id) cannot be cached; only single-expert groups go through the cache (otherwise the size overflows → cudaMemcpyAsync invalid argument crash)
2. **maybe_init position**: it was originally in the A3 slow pipeline, so the host buffer fast path never triggered it → moved to the function entry (unconditional execution)
3. **Cross-DSO call**: ggml-backend (libggml-base.so) cannot include CUDA headers directly; exposed via a proc-address registry (`ggml_cuda_expert_cache_copy_if_hit` / `set`)

## Code locations

- Cloud machine: `/root/llama.cpp-clean/ggml/src/ggml-backend.cpp`, `ggml-cuda/expert-cache.cuh/cu`, `ggml-cuda/ggml-cuda.cu`
- Backups: cloud machine `/root/moe-l2-backups/sched-cache-fix-20260802/`, local `测试数据备份/a3on-fix-20260802/`

---

## 2026-08-07 update: cache cap raised to 2048 slots (universal gain across three models)

> The 08-02 conclusion "Qwen/Mixtral no benefit, keep cache off" was overturned after EXPERT_CACHE_MAX_SLOTS 512 → 2048: insufficient capacity was the main cause of the original conclusion (small models have few experts, 512 slots were enough so no difference was measurable; V4 activates >512 experts per token, causing direct LRU thrashing).

### Three-model measurements (RTX 4090, 2026-08-07, cache=1.0)

| Model | No cache | + cache 2048 | Gain | GPU util change |
|------|---------|-------------|------|--------------|
| Qwen3.6-A3B | 46.9 / 44.8 | **50.2 / 49.8** | +7% / +11% | — |
| DS-V2-Lite | 36.4 | **37.9 / 37.2** | +4% | — |
| V4-Flash | 9.5 | **10.1** | +6% | 13% → 86% (near compute-bound) |

### Conclusions (2026-08-07)

1. **2048 slots is V4's sweet spot**: 512 slots gives no gain (activates >512 experts per token), 4096 slots OOMs (17.6GB cache + 8.4GB base > 24GB)
2. **Recommended config**: `GGML_CUDA_EXPERT_CACHE=1` (already built into cli.py), enable for all models
3. V4 GPU util at 86% is near compute-bound — the cache has solved the copy bottleneck; further speedup requires kernel/quantization optimization
4. Detailed troubleshooting chain and data: `/opt/data/moe-l2/历史记录文档/on-demand-pin-方案-交接-20260807.md`
